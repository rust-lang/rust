/**
 * A structure shared across REPL instances for storing history
 * such as statements and view items. I wish the AST was sendable.
 */
struct Repl {
    prompt: ~str,
    binary: ~str,
    running: bool,
    view_items: ~str,
    stmts: ~str
}

/// A utility function that hands off a pretty printer to a callback.
fn with_pp(intr: @token::ident_interner,
           cb: fn(pprust::ps, io::Writer)) -> ~str {
    do io::with_str_writer |writer| {
        let pp = pprust::rust_printer(writer, intr);

        cb(pp, writer);
        pp::eof(pp.s);
    }
}

/**
 * The AST (or the rest of rustc) are not sendable yet,
 * so recorded things are printed to strings. A terrible hack that
 * needs changes to rustc in order to be outed. This is unfortunately
 * going to cause the REPL to regress in parser performance,
 * because it has to parse the statements and view_items on each
 * input.
 */
fn record(repl: Repl, blk: @ast::blk, intr: @token::ident_interner) -> Repl {
    let view_items = if blk.node.view_items.len() > 0 {
        let new_view_items = do with_pp(intr) |pp, writer| {
            for blk.node.view_items.each |view_item| {
                pprust::print_view_item(pp, *view_item);
                writer.write_line(~"");
            }
        };

        debug!("new view items %s", new_view_items);

        repl.view_items + "\n" + new_view_items
    } else { repl.view_items };
    let stmts = if blk.node.stmts.len() > 0 {
        let new_stmts = do with_pp(intr) |pp, writer| {
            for blk.node.stmts.each |stmt| {
                match stmt.node {
                    ast::stmt_decl(*) => {
                        pprust::print_stmt(pp, **stmt);
                        writer.write_line(~"");
                    }
                    ast::stmt_expr(expr, _) | ast::stmt_semi(expr, _) => {
                        match expr.node {
                            ast::expr_assign(*) |
                            ast::expr_assign_op(*) |
                            ast::expr_swap(*) => {
                                pprust::print_stmt(pp, **stmt);
                                writer.write_line(~"");
                            }
                            _ => {}
                        }
                    }
                }
            }
        };

        debug!("new stmts %s", new_stmts);

        repl.stmts + "\n" + new_stmts
    } else { repl.stmts };

    Repl{
        view_items: view_items,
        stmts: stmts,
        .. repl
    }
}

/// Run an input string in a Repl, returning the new Repl.
fn run(repl: Repl, input: ~str) -> Repl {
    let options: @session::options = @{
        crate_type: session::unknown_crate,
        binary: repl.binary,
        addl_lib_search_paths: ~[os::getcwd()],
        .. *session::basic_options()
    };

    debug!("building driver input");
    let head = include_str!("wrapper.rs");
    let foot = fmt!("%s\nfn main() {\n%s\n\nprint({\n%s\n})\n}",
                    repl.view_items, repl.stmts, input);
    let wrapped = driver::str_input(head + foot);

    debug!("inputting %s", head + foot);

    debug!("building a driver session");
    let sess = driver::build_session(options, diagnostic::emit);

    debug!("building driver configuration");
    let cfg = driver::build_configuration(sess,
                                          repl.binary,
                                          wrapped);

    debug!("parsing");
    let mut crate = driver::parse_input(sess, cfg, wrapped);
    let mut opt = None;

    for crate.node.module.items.each |item| {
        match item.node {
            ast::item_fn(_, _, _, blk) => {
                if item.ident == sess.ident_of(~"main") {
                    opt = blk.node.expr;
                }
            }
            _ => {}
        }
    }

    let blk = match opt.get().node {
        ast::expr_call(_, exprs, _) => {
            match exprs[0].node {
                ast::expr_block(blk) => @blk,
                _ => fail
            }
        }
        _ => fail
    };

    debug!("configuration");
    crate = front::config::strip_unconfigured_items(crate);

    debug!("maybe building test harness");
    crate = front::test::modify_for_testing(sess, crate);

    debug!("expansion");
    crate = syntax::ext::expand::expand_crate(sess.parse_sess,
                                              sess.opts.cfg,
                                              crate);

    debug!("intrinsic injection");
    crate = front::intrinsic_inject::inject_intrinsic(sess, crate);

    debug!("core injection");
    crate = front::core_inject::maybe_inject_libcore_ref(sess, crate);

    debug!("building lint settings table");
    lint::build_settings_crate(sess, crate);

    debug!("ast indexing");
    let ast_map = syntax::ast_map::map_crate(sess.diagnostic(), *crate);

    debug!("external crate/lib resolution");
    creader::read_crates(sess.diagnostic(), *crate, sess.cstore,
                         sess.filesearch,
                         session::sess_os_to_meta_os(sess.targ_cfg.os),
                         sess.opts.static, sess.parse_sess.interner);

    debug!("language item collection");
    let lang_items = middle::lang_items::collect_language_items(crate, sess);

    debug!("resolution");
    let {def_map: def_map,
         exp_map2: exp_map2,
         trait_map: trait_map} = middle::resolve::resolve_crate(sess,
                                                                lang_items,
                                                                crate);

    debug!("freevar finding");
    let freevars = freevars::annotate_freevars(def_map, crate);

    debug!("region_resolution");
    let region_map = middle::region::resolve_crate(sess, def_map, crate);

    debug!("region paramaterization inference");
    let rp_set = middle::region::determine_rp_in_crate(sess, ast_map,
                                                       def_map, crate);

    debug!("typechecking");
    let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars,
                            region_map, rp_set, move lang_items, crate);
    let (method_map, vtable_map) = typeck::check_crate(ty_cx, trait_map,
                                                       crate);

    debug!("const marking");
    middle::const_eval::process_crate(crate, def_map, ty_cx);

    debug!("const checking");
    middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                     method_map, ty_cx);

    debug!("privacy checking");
    middle::privacy::check_crate(ty_cx, &method_map, crate);

    debug!("loop checking");
    middle::check_loop::check_crate(ty_cx, crate);

    debug!("alt checking");
    middle::check_alt::check_crate(ty_cx, crate);

    debug!("liveness checking");
    let last_use_map = middle::liveness::check_crate(ty_cx,
                                                     method_map, crate);

    debug!("borrow checking");
    let (root_map, mutbl_map) = middle::borrowck::check_crate(ty_cx,
                                                              method_map,
                                                              last_use_map,
                                                              crate);

    debug!("kind checking");
    kind::check_crate(ty_cx, method_map, last_use_map, crate);

    debug!("lint checking");
    lint::check_crate(ty_cx, crate);

    let maps = {mutbl_map: mutbl_map,
                root_map: root_map,
                last_use_map: last_use_map,
                method_map: method_map,
                vtable_map: vtable_map};

    debug!("translation");
    let (llmod, _) = trans::base::trans_crate(sess, crate, ty_cx,
                                              ~path::from_str("<repl>"),
                                              exp_map2, maps);
    let pm = llvm::LLVMCreatePassManager();

    debug!("executing jit");
    back::link::jit::exec(sess, pm, llmod, 0, false);
    llvm::LLVMDisposePassManager(pm);

    debug!("recording input into repl history");
    record(repl, blk, sess.parse_sess.interner)
}

/// Run a command, e.g. :clear, :exit, etc.
fn run_cmd(repl: &mut Repl, _in: io::Reader, _out: io::Writer,
           cmd: ~str, _args: ~[~str]) {
    match cmd {
        ~"exit" => repl.running = false,
        ~"clear" => {
            repl.view_items = ~"";
            repl.stmts = ~"";

            // FIXME: Win32 version of linenoise doesn't have the required function
            //rl::clear();
        }
        ~"help" => {
            io::println(~":clear - clear the screen\n" +
                        ~":exit - exit from the repl\n" +
                        ~":help - show this message");
        }
        _ => io::println(~"unknown cmd: " + cmd)
    }
}

fn main() {
    let args = os::args();
    let in = io::stdin();
    let out = io::stdout();
    let mut repl = Repl {
        prompt: ~"rusti> ",
        binary: args[0],
        running: true,
        view_items: ~"",
        stmts: ~""
    };

    do rl::complete |line, suggest| {
        if line.starts_with(":") {
            suggest(~":clear");
            suggest(~":exit");
            suggest(~":help");
        }
    }

    while repl.running {
        let result = rl::read(repl.prompt);

        if result.is_none() {
            break;
        }

        let line = result.get();

        if line.is_empty() {
            io::println(~"()");

            loop;
        }

        rl::add_history(line);

        if line.starts_with(~":") {
            let full = line.substr(1, line.len() - 1);
            let split = full.split_char(' ');
            let len = split.len();

            if len > 0 {
                let cmd = split[0];

                if !cmd.is_empty() {
                    let args = if len > 1 {
                        do vec::view(split, 1, len - 1).map |arg| {
                            *arg
                        }
                    } else { ~[] };

                    run_cmd(&mut repl, in, out, cmd, args);

                    loop;
                }
            }
        }

        let result = do task::try |copy repl| {
            run(copy repl, line)
        };

        if result.is_ok() {
            repl = result.get();
        }
    }
}
