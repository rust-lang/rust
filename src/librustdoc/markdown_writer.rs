// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use doc::ItemUtils;
use io::ReaderUtil;
use std::future;

pub enum WriteInstr {
    Write(~str),
    Done
}

pub type Writer = fn~(+v: WriteInstr);
pub type WriterFactory = fn~(+page: doc::Page) -> Writer;

pub trait WriterUtils {
    fn write_str(+str: ~str);
    fn write_line(+str: ~str);
    fn write_done();
}

impl Writer: WriterUtils {
    fn write_str(+str: ~str) {
        self(Write(str));
    }

    fn write_line(+str: ~str) {
        self.write_str(str + ~"\n");
    }

    fn write_done() {
        self(Done)
    }
}

pub fn make_writer_factory(+config: config::Config) -> WriterFactory {
    match config.output_format {
      config::Markdown => {
        markdown_writer_factory(config)
      }
      config::PandocHtml => {
        pandoc_writer_factory(config)
      }
    }
}

fn markdown_writer_factory(+config: config::Config) -> WriterFactory {
    fn~(+page: doc::Page) -> Writer {
        markdown_writer(config, page)
    }
}

fn pandoc_writer_factory(+config: config::Config) -> WriterFactory {
    fn~(+page: doc::Page) -> Writer {
        pandoc_writer(config, page)
    }
}

fn markdown_writer(
    +config: config::Config,
    +page: doc::Page
) -> Writer {
    let filename = make_local_filename(config, page);
    do generic_writer |markdown| {
        write_file(&filename, markdown);
    }
}

fn pandoc_writer(
    +config: config::Config,
    +page: doc::Page
) -> Writer {
    assert config.pandoc_cmd.is_some();
    let pandoc_cmd = config.pandoc_cmd.get();
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
        use io::WriterUtil;

        debug!("pandoc cmd: %s", pandoc_cmd);
        debug!("pandoc args: %s", str::connect(pandoc_args, ~" "));

        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();
        let pid = run::spawn_process(
            pandoc_cmd, pandoc_args, &None, &None,
            pipe_in.in, pipe_out.out, pipe_err.out);

        let writer = io::fd_writer(pipe_in.out, false);
        writer.write_str(markdown);

        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);
        os::close(pipe_in.out);

        let (stdout_po, stdout_ch) = pipes::stream();
        do task::spawn_sched(task::SingleThreaded) |move stdout_ch| {
            stdout_ch.send(readclose(pipe_out.in));
        }

        let (stderr_po, stderr_ch) = pipes::stream();
        do task::spawn_sched(task::SingleThreaded) |move stderr_ch| {
            stderr_ch.send(readclose(pipe_err.in));
        }
        let stdout = stdout_po.recv();
        let stderr = stderr_po.recv();

        let status = run::waitpid(pid);
        debug!("pandoc result: %i", status);
        if status != 0 {
            error!("pandoc-out: %s", stdout);
            error!("pandoc-err: %s", stderr);
            fail ~"pandoc failed";
        }
    }
}

fn readclose(fd: libc::c_int) -> ~str {
    // Copied from run::program_output
    let file = os::fdopen(fd);
    let reader = io::FILE_reader(file, false);
    let buf = io::with_bytes_writer(|writer| {
        let mut bytes = [mut 0, ..4096];
        while !reader.eof() {
            let nread = reader.read(bytes, bytes.len());
            writer.write(bytes.view(0, nread));
        }
    });
    os::fclose(file);
    str::from_bytes(buf)
}

fn generic_writer(+process: fn~(+markdown: ~str)) -> Writer {
    let (setup_po, setup_ch) = pipes::stream();
    do task::spawn |move process, move setup_ch| {
        let po: oldcomm::Port<WriteInstr> = oldcomm::Port();
        let ch = oldcomm::Chan(&po);
        setup_ch.send(ch);

        let mut markdown = ~"";
        let mut keep_going = true;
        while keep_going {
            match po.recv() {
              Write(s) => markdown += s,
              Done => keep_going = false
            }
        }
        process(move markdown);
    };
    let ch = setup_po.recv();

    fn~(+instr: WriteInstr) {
        oldcomm::send(ch, instr);
    }
}

fn make_local_filename(
    +config: config::Config,
    +page: doc::Page
) -> Path {
    let filename = make_filename(config, page);
    config.output_dir.push_rel(&filename)
}

pub fn make_filename(
    +config: config::Config,
    +page: doc::Page
) -> Path {
    let filename = {
        match page {
          doc::CratePage(doc) => {
            if config.output_format == config::PandocHtml &&
                config.output_style == config::DocPerMod {
                ~"index"
            } else {
                assert doc.topmod.name() != ~"";
                doc.topmod.name()
            }
          }
          doc::ItemPage(doc) => {
            str::connect(doc.path() + ~[doc.name()], ~"_")
          }
        }
    };
    let ext = match config.output_format {
      config::Markdown => ~"md",
      config::PandocHtml => ~"html"
    };

    Path(filename).with_filetype(ext)
}

#[test]
fn should_use_markdown_file_name_based_off_crate() {
    let config = {
        output_dir: Path("output/dir"),
        output_format: config::Markdown,
        output_style: config::DocPerCrate,
        .. config::default_config(&Path("input/test.rc"))
    };
    let doc = test::mk_doc(~"test", ~"");
    let page = doc::CratePage(doc.CrateDoc());
    let filename = make_local_filename(config, page);
    assert filename.to_str() == ~"output/dir/test.md";
}

#[test]
fn should_name_html_crate_file_name_index_html_when_doc_per_mod() {
    let config = {
        output_dir: Path("output/dir"),
        output_format: config::PandocHtml,
        output_style: config::DocPerMod,
        .. config::default_config(&Path("input/test.rc"))
    };
    let doc = test::mk_doc(~"", ~"");
    let page = doc::CratePage(doc.CrateDoc());
    let filename = make_local_filename(config, page);
    assert filename.to_str() == ~"output/dir/index.html";
}

#[test]
fn should_name_mod_file_names_by_path() {
    let config = {
        output_dir: Path("output/dir"),
        output_format: config::PandocHtml,
        output_style: config::DocPerMod,
        .. config::default_config(&Path("input/test.rc"))
    };
    let doc = test::mk_doc(~"", ~"mod a { mod b { } }");
    let modb = doc.cratemod().mods()[0].mods()[0];
    let page = doc::ItemPage(doc::ModTag(modb));
    let filename = make_local_filename(config, page);
    assert  filename == Path("output/dir/a_b.html");
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc(+name: ~str, +source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, name);
            let doc = (path_pass::mk_pass().f)(srv, doc);
            doc
        }
    }
}

fn write_file(path: &Path, +s: ~str) {
    use io::WriterUtil;

    match io::file_writer(path, ~[io::Create, io::Truncate]) {
      result::Ok(writer) => {
        writer.write_str(s);
      }
      result::Err(e) => fail e
    }
}

pub fn future_writer_factory(
) -> (WriterFactory, oldcomm::Port<(doc::Page, ~str)>) {
    let markdown_po = oldcomm::Port();
    let markdown_ch = oldcomm::Chan(&markdown_po);
    let writer_factory = fn~(+page: doc::Page) -> Writer {
        let (writer_po, writer_ch) = pipes::stream();
        do task::spawn |move writer_ch| {
            let (writer, future) = future_writer();
            writer_ch.send(move writer);
            let s = future.get();
            oldcomm::send(markdown_ch, (page, s));
        }
        writer_po.recv()
    };

    (move writer_factory, markdown_po)
}

fn future_writer() -> (Writer, future::Future<~str>) {
    let (port, chan) = pipes::stream();
    let writer = fn~(move chan, +instr: WriteInstr) {
        chan.send(copy instr);
    };
    let future = do future::from_fn |move port| {
        let mut res = ~"";
        loop {
            match port.recv() {
              Write(s) => res += s,
              Done => break
            }
        }
        res
    };
    (move writer, move future)
}
