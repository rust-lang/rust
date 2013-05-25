// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use config;
use doc::ItemUtils;
use doc;

use core::comm::*;
use core::comm;
use core::io;
use core::libc;
use core::os;
use core::result;
use core::run;
use core::str;
use core::task;
use extra::future;

pub enum WriteInstr {
    Write(~str),
    Done
}

pub type Writer = ~fn(v: WriteInstr);
pub type WriterFactory = ~fn(page: doc::Page) -> Writer;

pub trait WriterUtils {
    fn put_str(&self, str: ~str);
    fn put_line(&self, str: ~str);
    fn put_done(&self);
}

impl WriterUtils for Writer {
    fn put_str(&self, str: ~str) {
        (*self)(Write(str));
    }

    fn put_line(&self, str: ~str) {
        self.put_str(str + "\n");
    }

    fn put_done(&self) {
        (*self)(Done)
    }
}

pub fn make_writer_factory(config: config::Config) -> WriterFactory {
    match config.output_format {
      config::Markdown => {
        markdown_writer_factory(config)
      }
      config::PandocHtml => {
        pandoc_writer_factory(config)
      }
    }
}

fn markdown_writer_factory(config: config::Config) -> WriterFactory {
    let result: ~fn(page: doc::Page) -> Writer = |page| {
        markdown_writer(&config, page)
    };
    result
}

fn pandoc_writer_factory(config: config::Config) -> WriterFactory {
    let result: ~fn(doc::Page) -> Writer = |page| {
        pandoc_writer(&config, page)
    };
    result
}

fn markdown_writer(
    config: &config::Config,
    page: doc::Page
) -> Writer {
    let filename = make_local_filename(config, page);
    do generic_writer |markdown| {
        write_file(&filename, markdown);
    }
}

fn pandoc_writer(
    config: &config::Config,
    page: doc::Page
) -> Writer {
    assert!(config.pandoc_cmd.is_some());
    let pandoc_cmd = copy *config.pandoc_cmd.get_ref();
    let filename = make_local_filename(config, page);

    let pandoc_args = ~[
        ~"--standalone",
        ~"--section-divs",
        ~"--from=markdown",
        ~"--to=html",
        ~"--css=rust.css",
        ~"--output=" + filename.to_str()
    ];

    do generic_writer |markdown| {
        use core::io::WriterUtil;

        debug!("pandoc cmd: %s", pandoc_cmd);
        debug!("pandoc args: %s", str::connect(pandoc_args, " "));

        let mut proc = run::Process::new(pandoc_cmd, pandoc_args, run::ProcessOptions::new());

        proc.input().write_str(markdown);
        let output = proc.finish_with_output();

        debug!("pandoc result: %i", output.status);
        if output.status != 0 {
            error!("pandoc-out: %s", str::from_bytes(output.output));
            error!("pandoc-err: %s", str::from_bytes(output.error));
            fail!("pandoc failed");
        }
    }
}

fn generic_writer(process: ~fn(markdown: ~str)) -> Writer {
    let (po, ch) = stream::<WriteInstr>();
    do task::spawn || {
        let mut markdown = ~"";
        let mut keep_going = true;
        while keep_going {
            match po.recv() {
              Write(s) => markdown += s,
              Done => keep_going = false
            }
        }
        process(markdown);
    };
    let result: ~fn(instr: WriteInstr) = |instr| ch.send(instr);
    result
}

pub fn make_local_filename(
    config: &config::Config,
    page: doc::Page
) -> Path {
    let filename = make_filename(config, page);
    config.output_dir.push_rel(&filename)
}

pub fn make_filename(
    config: &config::Config,
    page: doc::Page
) -> Path {
    let filename = {
        match page {
          doc::CratePage(doc) => {
            if config.output_format == config::PandocHtml &&
                config.output_style == config::DocPerMod {
                ~"index"
            } else {
                assert!(doc.topmod.name() != ~"");
                doc.topmod.name()
            }
          }
          doc::ItemPage(doc) => {
            str::connect(doc.path() + [doc.name()], "_")
          }
        }
    };
    let ext = match config.output_format {
      config::Markdown => ~"md",
      config::PandocHtml => ~"html"
    };

    Path(filename).with_filetype(ext)
}

fn write_file(path: &Path, s: ~str) {
    use core::io::WriterUtil;

    match io::file_writer(path, [io::Create, io::Truncate]) {
      result::Ok(writer) => {
        writer.write_str(s);
      }
      result::Err(e) => fail!(e)
    }
}

pub fn future_writer_factory(
) -> (WriterFactory, Port<(doc::Page, ~str)>) {
    let (markdown_po, markdown_ch) = stream();
    let markdown_ch = SharedChan::new(markdown_ch);
    let writer_factory: WriterFactory = |page| {
        let (writer_po, writer_ch) = comm::stream();
        let markdown_ch = markdown_ch.clone();
        do task::spawn || {
            let (writer, future) = future_writer();
            let mut future = future;
            writer_ch.send(writer);
            let s = future.get();
            markdown_ch.send((copy page, s));
        }
        writer_po.recv()
    };

    (writer_factory, markdown_po)
}

fn future_writer() -> (Writer, future::Future<~str>) {
    let (port, chan) = comm::stream();
    let writer: ~fn(instr: WriteInstr) = |instr| chan.send(copy instr);
    let future = do future::from_fn || {
        let mut res = ~"";
        loop {
            match port.recv() {
              Write(s) => res += s,
              Done => break
            }
        }
        res
    };
    (writer, future)
}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use astsrv;
    use doc;
    use extract;
    use path_pass;
    use config;
    use super::make_local_filename;

    fn mk_doc(name: ~str, source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv.clone(), copy name);
            let doc = (path_pass::mk_pass().f)(srv.clone(), doc);
            doc
        }
    }

    #[test]
    fn should_use_markdown_file_name_based_off_crate() {
        let config = config::Config {
            output_dir: Path("output/dir"),
            output_format: config::Markdown,
            output_style: config::DocPerCrate,
            .. config::default_config(&Path("input/test.rc"))
        };
        let doc = mk_doc(~"test", ~"");
        let page = doc::CratePage(doc.CrateDoc());
        let filename = make_local_filename(&config, page);
        assert_eq!(filename.to_str(), ~"output/dir/test.md");
    }

    #[test]
    fn should_name_html_crate_file_name_index_html_when_doc_per_mod() {
        let config = config::Config {
            output_dir: Path("output/dir"),
            output_format: config::PandocHtml,
            output_style: config::DocPerMod,
            .. config::default_config(&Path("input/test.rc"))
        };
        let doc = mk_doc(~"", ~"");
        let page = doc::CratePage(doc.CrateDoc());
        let filename = make_local_filename(&config, page);
        assert_eq!(filename.to_str(), ~"output/dir/index.html");
    }

    #[test]
    fn should_name_mod_file_names_by_path() {
        let config = config::Config {
            output_dir: Path("output/dir"),
            output_format: config::PandocHtml,
            output_style: config::DocPerMod,
            .. config::default_config(&Path("input/test.rc"))
        };
        let doc = mk_doc(~"", ~"mod a { mod b { } }");
        let modb = copy doc.cratemod().mods()[0].mods()[0];
        let page = doc::ItemPage(doc::ModTag(modb));
        let filename = make_local_filename(&config, page);
        assert_eq!(filename, Path("output/dir/a_b.html"));
    }
}
