// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;
use std::hashmap::HashMap;
use std::local_data;
use std::sys;

use syntax::ast;
use syntax::parse::token;
use syntax::print::pprust;
use rustc::middle::ty;
use rustc::util::ppaux;

use utils::*;

/// This structure keeps track of the state of the world for the code being
/// executed in rusti.
struct Program {
    /// All known local variables
    local_vars: HashMap<~str, LocalVariable>,
    /// New variables which will be present (learned from typechecking)
    newvars: HashMap<~str, LocalVariable>,
    /// All known view items (use statements), distinct because these must
    /// follow extern mods
    view_items: ~str,
    /// All known 'extern mod' statements (must always come first)
    externs: ~str,
    /// All known structs defined. These need to have
    /// #[deriving(Encodable,Decodable)] to be at all useful in rusti
    structs: HashMap<~str, ~str>,
    /// All other items, can all be intermingled. Duplicate definitions of the
    /// same name have the previous one overwritten.
    items: HashMap<~str, ~str>,
}

/// Represents a local variable that the program is currently using.
struct LocalVariable {
    /// Should this variable be locally declared as mutable?
    mutable: bool,
    /// This is the type of the serialized data below
    ty: ~str,
    /// This is the serialized version of the variable
    data: ~[u8],
    /// When taking borrowed pointers or slices, care must be taken to ensure
    /// that the deserialization produces what we'd expect. If some magic is in
    /// order, the first element of this pair is the actual type of the local
    /// variable (which can be different from the deserialized type), and the
    /// second element are the '&'s which need to be prepended.
    alterations: Option<(~str, ~str)>,
}

type LocalCache = @mut HashMap<~str, @~[u8]>;
fn tls_key(_k: @LocalCache) {}

impl Program {
    pub fn new() -> Program {
        Program {
            local_vars: HashMap::new(),
            newvars: HashMap::new(),
            view_items: ~"",
            externs: ~"",
            structs: HashMap::new(),
            items: HashMap::new(),
        }
    }

    /// Clears all local bindings about variables, items, externs, etc.
    pub fn clear(&mut self) {
        *self = Program::new();
    }

    /// Creates a block of code to be fed to rustc. This code is not meant to
    /// run, but rather it is meant to learn about the input given. This will
    /// assert that the types of all bound local variables are encodable,
    /// along with checking syntax and other rust-related things. The reason
    /// that we only check for encodability is that some super-common types
    /// (like &'static str) are not decodable, but are encodable. By doing some
    /// mild approximation when decoding, we can emulate at least &str and &[T].
    ///
    /// Once this code has been fed to rustc, it is intended that the code()
    /// function is used to actually generate code to fully compile and run.
    pub fn test_code(&self, user_input: &str, to_print: &Option<~str>,
                     new_locals: &[(~str, bool)]) -> ~str {
        let mut code = self.program_header();
        code.push_str("
    fn assert_encodable<T: Encodable<::extra::ebml::writer::Encoder>>(t: &T) {}
        ");

        code.push_str("fn main() {\n");
        // It's easy to initialize things if we don't run things...
        for self.local_vars.each |name, var| {
            let mt = var.mt();
            code.push_str(fmt!("let%s %s: %s = fail!();\n", mt, *name, var.ty));
            var.alter(*name, &mut code);
        }
        code.push_str("{\n");
        code.push_str(user_input);
        code.push_char('\n');
        match *to_print {
            Some(ref s) => {
                code.push_str(*s);
                code.push_char('\n');
            }
            None => {}
        }

        for new_locals.each |p| {
            code.push_str(fmt!("assert_encodable(&%s);\n", *p.first_ref()));
        }
        code.push_str("};}");
        return code;
    }

    /// Creates a program to be fed into rustc. This program is structured to
    /// deserialize all bindings into local variables, run the code input, and
    /// then reserialize all the variables back out.
    ///
    /// This program (unlike test_code) is meant to run to actually execute the
    /// user's input
    pub fn code(&mut self, user_input: &str, to_print: &Option<~str>) -> ~str {
        let mut code = self.program_header();
        code.push_str("
            fn main() {
        ");

        let key: sys::Closure = unsafe {
            let tls_key: &'static fn(@LocalCache) = tls_key;
            cast::transmute(tls_key)
        };
        // First, get a handle to the tls map which stores all the local
        // variables. This works by totally legitimately using the 'code'
        // pointer of the 'tls_key' function as a uint, and then casting it back
        // up to a function
        code.push_str(fmt!("
            let __tls_map: @mut ::std::hashmap::HashMap<~str, @~[u8]> = unsafe {
                let key = ::std::sys::Closure{ code: %? as *(),
                                               env: ::std::ptr::null() };
                let key = ::std::cast::transmute(key);
                *::std::local_data::local_data_get(key).unwrap()
            };\n", key.code as uint));

        // Using this __tls_map handle, deserialize each variable binding that
        // we know about
        for self.local_vars.each |name, var| {
            let mt = var.mt();
            code.push_str(fmt!("let%s %s: %s = {
                let data = __tls_map.get_copy(&~\"%s\");
                let doc = ::extra::ebml::reader::Doc(data);
                let mut decoder = ::extra::ebml::reader::Decoder(doc);
                ::extra::serialize::Decodable::decode(&mut decoder)
            };\n", mt, *name, var.ty, *name));
            var.alter(*name, &mut code);
        }

        // After all that, actually run the user's code.
        code.push_str(user_input);
        code.push_char('\n');

        match *to_print {
            Some(ref s) => { code.push_str(fmt!("pp({\n%s\n});", *s)); }
            None => {}
        }

        do self.newvars.consume |name, var| {
            self.local_vars.insert(name, var);
        }

        // After the input code is run, we can re-serialize everything back out
        // into tls map (to be read later on by this task)
        for self.local_vars.each |name, var| {
            code.push_str(fmt!("{
                let local: %s = %s;
                let bytes = do ::std::io::with_bytes_writer |io| {
                    let mut enc = ::extra::ebml::writer::Encoder(io);
                    local.encode(&mut enc);
                };
                __tls_map.insert(~\"%s\", @bytes);
            }\n", var.real_ty(), *name, *name));
        }

        // Close things up, and we're done.
        code.push_str("}");
        return code;
    }

    /// Creates the header of the programs which are generated to send to rustc
    fn program_header(&self) -> ~str {
        // up front, disable lots of annoying lints, then include all global
        // state such as items, view items, and extern mods.
        let mut code = fmt!("
            #[allow(ctypes)];
            #[allow(heap_memory)];
            #[allow(implicit_copies)];
            #[allow(managed_heap_memory)];
            #[allow(non_camel_case_types)];
            #[allow(owned_heap_memory)];
            #[allow(path_statement)];
            #[allow(unrecognized_lint)];
            #[allow(unused_imports)];
            #[allow(while_true)];
            #[allow(unused_variable)];
            #[allow(dead_assignment)];
            #[allow(unused_unsafe)];
            #[allow(unused_mut)];
            #[allow(unreachable_code)];

            extern mod extra;
            %s // extern mods

            use extra::serialize::*;
            %s // view items
        ", self.externs, self.view_items);
        for self.structs.each_value |s| {
            // The structs aren't really useful unless they're encodable
            code.push_str("#[deriving(Encodable, Decodable)]");
            code.push_str(*s);
            code.push_str("\n");
        }
        for self.items.each_value |s| {
            code.push_str(*s);
            code.push_str("\n");
        }
        code.push_str("fn pp<T>(t: T) { println(fmt!(\"%?\", t)); }\n");
        return code;
    }

    /// Initializes the task-local cache of all local variables known to the
    /// program. This will be used to read local variables out of once the
    /// program starts
    pub fn set_cache(&self) {
        let map = @mut HashMap::new();
        for self.local_vars.each |name, value| {
            map.insert(copy *name, @copy value.data);
        }
        unsafe {
            local_data::local_data_set(tls_key, @map);
        }
    }

    /// Once the program has finished running, this function will consume the
    /// task-local cache of local variables. After the program finishes running,
    /// it updates this cache with the new values of each local variable.
    pub fn consume_cache(&mut self) {
        let map = unsafe {
            local_data::local_data_pop(tls_key).expect("tls is empty")
        };
        do map.consume |name, value| {
            match self.local_vars.find_mut(&name) {
                Some(v) => { v.data = copy *value; }
                None => { fail!("unknown variable %s", name) }
            }
        }
    }

    // Simple functions to record various global things (as strings)

    pub fn record_view_item(&mut self, vi: &str) {
        self.view_items.push_str(vi);
        self.view_items.push_char('\n');
    }

    pub fn record_struct(&mut self, name: &str, s: ~str) {
        let name = name.to_owned();
        self.items.remove(&name);
        self.structs.insert(name, s);
    }

    pub fn record_item(&mut self, name: &str, it: ~str) {
        let name = name.to_owned();
        self.structs.remove(&name);
        self.items.insert(name, it);
    }

    pub fn record_extern(&mut self, name: &str) {
        self.externs.push_str(name);
        self.externs.push_char('\n');
    }

    /// This monster function is responsible for reading the main function
    /// generated by test_code() to determine the type of each local binding
    /// created by the user's input.
    ///
    /// Once the types are known, they are inserted into the local_vars map in
    /// this Program (to be deserialized later on
    pub fn register_new_vars(&mut self, blk: &ast::blk, tcx: ty::ctxt) {
        debug!("looking for new variables");
        let newvars = @mut HashMap::new();
        do each_user_local(blk) |local| {
            let mutable = local.node.is_mutbl;
            do each_binding(local) |path, id| {
                let name = do with_pp(token::get_ident_interner()) |pp, _| {
                    pprust::print_path(pp, path, false);
                };
                let mut t = ty::node_id_to_type(tcx, id);
                let mut tystr = ~"";
                let mut lvar = LocalVariable {
                    ty: ~"",
                    data: ~[],
                    mutable: mutable,
                    alterations: None,
                };
                // This loop is responsible for figuring out what "alterations"
                // are necessary for this local variable.
                loop {
                    match ty::get(t).sty {
                        // &T encoded will decode to T, so we need to be sure to
                        // re-take a loan after decoding
                        ty::ty_rptr(_, mt) => {
                            if mt.mutbl == ast::m_mutbl {
                                tystr.push_str("&mut ");
                            } else {
                                tystr.push_str("&");
                            }
                            t = mt.ty;
                        }
                        // Literals like [1, 2, 3] and (~[0]).slice() will both
                        // be serialized to ~[T], whereas it's requested to be a
                        // &[T] instead.
                        ty::ty_evec(mt, ty::vstore_slice(*)) |
                        ty::ty_evec(mt, ty::vstore_fixed(*)) => {
                            let vty = ppaux::ty_to_str(tcx, mt.ty);
                            let derefs = copy tystr;
                            lvar.ty = tystr + "~[" + vty + "]";
                            lvar.alterations = Some((tystr + "&[" + vty + "]",
                                                     derefs));
                            break;
                        }
                        // Similar to vectors, &str serializes to ~str, so a
                        // borrow must be taken
                        ty::ty_estr(ty::vstore_slice(*)) => {
                            let derefs = copy tystr;
                            lvar.ty = tystr + "~str";
                            lvar.alterations = Some((tystr + "&str", derefs));
                            break;
                        }
                        // Don't generate extra stuff if there's no borrowing
                        // going on here
                        _ if "" == tystr => {
                            lvar.ty = ppaux::ty_to_str(tcx, t);
                            break;
                        }
                        // If we're just borrowing (no vectors or strings), then
                        // we just need to record how many borrows there were.
                        _ => {
                            let derefs = copy tystr;
                            let tmptystr = ppaux::ty_to_str(tcx, t);
                            lvar.alterations = Some((tystr + tmptystr, derefs));
                            lvar.ty = tmptystr;
                            break;
                        }
                    }
                }
                newvars.insert(name, lvar);
            }
        }

        // I'm not an @ pointer, so this has to be done outside.
        do newvars.consume |k, v| {
            self.newvars.insert(k, v);
        }

        // helper functions to perform ast iteration
        fn each_user_local(blk: &ast::blk, f: &fn(@ast::local)) {
            do find_user_block(blk) |blk| {
                for blk.node.stmts.each |stmt| {
                    match stmt.node {
                        ast::stmt_decl(d, _) => {
                            match d.node {
                                ast::decl_local(l) => { f(l); }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        fn find_user_block(blk: &ast::blk, f: &fn(&ast::blk)) {
            for blk.node.stmts.each |stmt| {
                match stmt.node {
                    ast::stmt_semi(e, _) => {
                        match e.node {
                            ast::expr_block(ref blk) => { return f(blk); }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            fail!("couldn't find user block");
        }
    }
}

impl LocalVariable {
    /// Performs alterations to the code provided, given the name of this
    /// variable.
    fn alter(&self, name: &str, code: &mut ~str) {
        match self.alterations {
            Some((ref real_ty, ref prefix)) => {
                code.push_str(fmt!("let%s %s: %s = %s%s;\n",
                                   self.mt(), name,
                                   *real_ty, *prefix, name));
            }
            None => {}
        }
    }

    fn real_ty<'a>(&'a self) -> &'a str {
        match self.alterations {
            Some((ref real_ty, _)) => {
                let ret: &'a str = *real_ty;
                return ret;
            }
            None => {
                let ret: &'a str = self.ty;
                return ret;
            }
        }
    }

    fn mt(&self) -> &'static str {
        if self.mutable {" mut"} else {""}
    }
}
