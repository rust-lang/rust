export writeinstr;
export writer;
export writer_factory;
export writer_util;
export make_writer_factory;
export future_writer_factory;

enum writeinstr {
    write(str),
    done
}

type writer = fn~(+writeinstr);
type writer_factory = fn~(page: doc::page) -> writer;

impl writer_util for writer {
    fn write_str(str: str) {
        self(write(str));
    }

    fn write_line(str: str) {
        self.write_str(str + "\n");
    }

    fn write_done() {
        self(done)
    }
}

fn make_writer_factory(config: config::config) -> writer_factory {
    alt config.output_format {
      config::markdown {
        markdown_writer_factory(config)
      }
      config::pandoc_html {
        pandoc_writer_factory(config)
      }
    }
}

fn markdown_writer_factory(config: config::config) -> writer_factory {
    fn~(page: doc::page) -> writer {
        markdown_writer(config, page)
    }
}

fn pandoc_writer_factory(config: config::config) -> writer_factory {
    fn~(page: doc::page) -> writer {
        pandoc_writer(config, page)
    }
}

fn markdown_writer(
    config: config::config,
    page: doc::page
) -> writer {
    let filename = make_filename(config, "md", page);
    generic_writer {|markdown|
        write_file(filename, markdown);
    }
}

fn pandoc_writer(
    config: config::config,
    page: doc::page
) -> writer {
    assert option::is_some(config.pandoc_cmd);
    let pandoc_cmd = option::get(config.pandoc_cmd);
    let filename = make_filename(config, "html", page);

    let pandoc_args = [
        "--standalone",
        "--section-divs",
        "--from=markdown",
        "--to=html",
        "--css=rust.css",
        "--output=" + filename
    ];

    generic_writer {|markdown|
        import std::run;
        import std::os;
        import std::io;
        import std::io::writer_util;

        #debug("pandoc cmd: %s", pandoc_cmd);
        #debug("pandoc args: %s", str::connect(pandoc_args, " "));

        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();
        let pid = run::spawn_process(
            pandoc_cmd, pandoc_args, none, none,
            pipe_in.in, pipe_out.out, pipe_err.out);

        let writer = io::fd_writer(pipe_in.out, false);
        writer.write_str(markdown);

        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);
        os::close(pipe_in.out);
        os::close(pipe_out.in);
        os::close(pipe_err.in);

        let status = run::waitpid(pid);
        #debug("pandoc result: %i", status);
        if status != 0 {
            fail "pandoc failed";
        }
    }
}

fn generic_writer(process: fn~(markdown: str)) -> writer {
    let ch = task::spawn_listener {|po: comm::port<writeinstr>|
        let markdown = "";
        let keep_going = true;
        while keep_going {
            alt comm::recv(po) {
              write(s) { markdown += s; }
              done { keep_going = false; }
            }
        }
        process(markdown);
    };

    fn~(+instr: writeinstr) {
        comm::send(ch, instr);
    }
}

fn make_filename(
    config: config::config,
    ext: str,
    _page: doc::page
) -> str {
    import std::fs;
    let cratefile = fs::basename(config.input_crate);
    let cratename = tuple::first(fs::splitext(cratefile));
    fs::connect(config.output_dir, cratename + "." + ext)
}

fn write_file(path: str, s: str) {
    import std::io;
    import std::io::writer_util;

    alt io::file_writer(path, [io::create, io::truncate]) {
      result::ok(writer) {
        writer.write_str(s);
      }
      result::err(e) { fail e }
    }
}

#[test]
fn should_use_markdown_file_name_based_off_crate() {
    let config = {
        output_dir: "output/dir"
        with config::default_config("input/test.rc")
    };
    let doc = test::mk_doc("");
    let page = doc::cratepage(doc.cratedoc());
    assert make_filename(config, "md", page) == "output/dir/test.md";
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            extract::from_srv(srv, "")
        }
    }
}

fn future_writer_factory(
) -> (writer_factory, comm::port<(doc::page, str)>) {
    let markdown_po = comm::port();
    let markdown_ch = comm::chan(markdown_po);
    let writer_factory = fn~(page: doc::page) -> writer {
        let writer_po = comm::port();
        let writer_ch = comm::chan(writer_po);
        task::spawn {||
            let (writer, future) = future_writer();
            comm::send(writer_ch, writer);
            let s = future::get(future);
            comm::send(markdown_ch, (page, s));
        }
        comm::recv(writer_po)
    };

    (writer_factory, markdown_po)
}

fn future_writer() -> (writer, future::future<str>) {
    let port = comm::port();
    let chan = comm::chan(port);
    let writer = fn~(+instr: writeinstr) {
        comm::send(chan, copy instr);
    };
    let future = future::from_fn {||
        let res = "";
        while true {
            alt comm::recv(port) {
              write(s) { res += s }
              done { break }
            }
        }
        res
    };
    (writer, future)
}
