// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::local_data;
use std::rt::io;

use syntax::ast;

use clean;
use html::render::{cache_key, current_location_key};

pub struct VisSpace(Option<ast::visibility>);
pub struct Method<'self>(&'self clean::SelfTy, &'self clean::FnDecl);

impl fmt::Default for clean::Generics {
    fn fmt(g: &clean::Generics, f: &mut fmt::Formatter) {
        if g.lifetimes.len() == 0 && g.type_params.len() == 0 { return }
        f.buf.write("&lt;".as_bytes());

        for (i, life) in g.lifetimes.iter().enumerate() {
            if i > 0 { f.buf.write(", ".as_bytes()); }
            write!(f.buf, "{}", *life);
        }

        if g.type_params.len() > 0 {
            if g.lifetimes.len() > 0 { f.buf.write(", ".as_bytes()); }

            for (i, tp) in g.type_params.iter().enumerate() {
                if i > 0 { f.buf.write(", ".as_bytes()) }
                f.buf.write(tp.name.as_bytes());

                if tp.bounds.len() > 0 {
                    f.buf.write(": ".as_bytes());
                    for (i, bound) in tp.bounds.iter().enumerate() {
                        if i > 0 { f.buf.write(" + ".as_bytes()); }
                        write!(f.buf, "{}", *bound);
                    }
                }
            }
        }
        f.buf.write("&gt;".as_bytes());
    }
}

impl fmt::Default for clean::Lifetime {
    fn fmt(l: &clean::Lifetime, f: &mut fmt::Formatter) {
        f.buf.write("'".as_bytes());
        f.buf.write(l.as_bytes());
    }
}

impl fmt::Default for clean::TyParamBound {
    fn fmt(bound: &clean::TyParamBound, f: &mut fmt::Formatter) {
        match *bound {
            clean::RegionBound => {
                f.buf.write("'static".as_bytes())
            }
            clean::TraitBound(ref ty) => {
                write!(f.buf, "{}", *ty);
            }
        }
    }
}

impl fmt::Default for clean::Path {
    fn fmt(path: &clean::Path, f: &mut fmt::Formatter) {
        if path.global { f.buf.write("::".as_bytes()) }
        for (i, seg) in path.segments.iter().enumerate() {
            if i > 0 { f.buf.write("::".as_bytes()) }
            f.buf.write(seg.name.as_bytes());

            if seg.lifetime.is_some() || seg.types.len() > 0 {
                f.buf.write("&lt;".as_bytes());
                match seg.lifetime {
                    Some(ref lifetime) => write!(f.buf, "{}", *lifetime),
                    None => {}
                }
                for (i, ty) in seg.types.iter().enumerate() {
                    if i > 0 || seg.lifetime.is_some() {
                        f.buf.write(", ".as_bytes());
                    }
                    write!(f.buf, "{}", *ty);
                }
                f.buf.write("&gt;".as_bytes());
            }
        }
    }
}

fn resolved_path(w: &mut io::Writer, id: ast::NodeId, path: &clean::Path) {
    // The generics will get written to both the title and link
    let mut generics = ~"";
    let last = path.segments.last();
    if last.lifetime.is_some() || last.types.len() > 0 {
        generics.push_str("&lt;");
        match last.lifetime {
            Some(ref lifetime) => generics.push_str(format!("{}", *lifetime)),
            None => {}
        }
        for (i, ty) in last.types.iter().enumerate() {
            if i > 0 || last.lifetime.is_some() {
                generics.push_str(", ");
            }
            generics.push_str(format!("{}", *ty));
        }
        generics.push_str("&gt;");
    }

    // Did someone say rightward-drift?
    do local_data::get(current_location_key) |loc| {
        let loc = loc.unwrap();
        do local_data::get(cache_key) |cache| {
            do cache.unwrap().read |cache| {
                match cache.paths.find(&id) {
                    // This is a documented path, link to it!
                    Some(&(ref fqp, shortty)) => {
                        let fqn = fqp.connect("::");
                        let mut same = 0;
                        for (a, b) in loc.iter().zip(fqp.iter()) {
                            if *a == *b {
                                same += 1;
                            } else {
                                break;
                            }
                        }

                        let mut url = ~"";
                        for _ in range(same, loc.len()) {
                            url.push_str("../");
                        }
                        if same == fqp.len() {
                            url.push_str(shortty);
                            url.push_str(".");
                            url.push_str(*fqp.last());
                            url.push_str(".html");
                        } else {
                            let remaining = fqp.slice_from(same);
                            let to_link = remaining.slice_to(remaining.len() - 1);
                            for component in to_link.iter() {
                                url.push_str(*component);
                                url.push_str("/");
                            }
                            url.push_str(shortty);
                            url.push_str(".");
                            url.push_str(*remaining.last());
                            url.push_str(".html");
                        }

                        write!(w, "<a class='{}' href='{}' title='{}'>{}</a>{}",
                               shortty, url, fqn, last.name, generics);
                    }
                    None => {
                        write!(w, "{}{}", last.name, generics);
                    }
                };
            }
        }
    }
}

impl fmt::Default for clean::Type {
    fn fmt(g: &clean::Type, f: &mut fmt::Formatter) {
        match *g {
            clean::TyParamBinder(id) | clean::Generic(id) => {
                do local_data::get(cache_key) |cache| {
                    do cache.unwrap().read |m| {
                        f.buf.write(m.typarams.get(&id).as_bytes());
                    }
                }
            }
            clean::Unresolved(*) => unreachable!(),
            clean::ResolvedPath{id, typarams: ref typarams, path: ref path} => {
                resolved_path(f.buf, id, path);
                match *typarams {
                    Some(ref params) => {
                        f.buf.write("&lt;".as_bytes());
                        for (i, param) in params.iter().enumerate() {
                            if i > 0 { f.buf.write(", ".as_bytes()) }
                            write!(f.buf, "{}", *param);
                        }
                        f.buf.write("&gt;".as_bytes());
                    }
                    None => {}
                }
            }
            // XXX: this should be a link
            clean::External(ref a, _) => {
                write!(f.buf, "{}", *a);
            }
            clean::Self(*) => f.buf.write("Self".as_bytes()),
            clean::Primitive(prim) => {
                let s = match prim {
                    ast::ty_int(ast::ty_i) => "int",
                    ast::ty_int(ast::ty_i8) => "i8",
                    ast::ty_int(ast::ty_i16) => "i16",
                    ast::ty_int(ast::ty_i32) => "i32",
                    ast::ty_int(ast::ty_i64) => "i64",
                    ast::ty_uint(ast::ty_u) => "uint",
                    ast::ty_uint(ast::ty_u8) => "u8",
                    ast::ty_uint(ast::ty_u16) => "u16",
                    ast::ty_uint(ast::ty_u32) => "u32",
                    ast::ty_uint(ast::ty_u64) => "u64",
                    ast::ty_float(ast::ty_f) => "float",
                    ast::ty_float(ast::ty_f32) => "f32",
                    ast::ty_float(ast::ty_f64) => "f64",
                    ast::ty_str => "str",
                    ast::ty_bool => "bool",
                    ast::ty_char => "char",
                };
                f.buf.write(s.as_bytes());
            }
            clean::Closure(ref decl) => {
                f.buf.write(match decl.sigil {
                    ast::BorrowedSigil => "&amp;",
                    ast::ManagedSigil => "@",
                    ast::OwnedSigil => "~",
                }.as_bytes());
                match decl.region {
                    Some(ref region) => write!(f.buf, "{} ", *region),
                    None => {}
                }
                write!(f.buf, "{}{}fn{}",
                       match decl.purity {
                           ast::unsafe_fn => "unsafe ",
                           ast::extern_fn => "extern ",
                           ast::impure_fn => ""
                       },
                       match decl.onceness {
                           ast::Once => "once ",
                           ast::Many => "",
                       },
                       decl.decl);
                // XXX: where are bounds and lifetimes printed?!
            }
            clean::BareFunction(ref decl) => {
                write!(f.buf, "{}{}fn{}{}",
                       match decl.purity {
                           ast::unsafe_fn => "unsafe ",
                           ast::extern_fn => "extern ",
                           ast::impure_fn => ""
                       },
                       match decl.abi {
                           ~"" | ~"\"Rust\"" => ~"",
                           ref s => " " + *s + " ",
                       },
                       decl.generics,
                       decl.decl);
            }
            clean::Tuple(ref typs) => {
                f.buf.write("(".as_bytes());
                for (i, typ) in typs.iter().enumerate() {
                    if i > 0 { f.buf.write(", ".as_bytes()) }
                    write!(f.buf, "{}", *typ);
                }
                f.buf.write(")".as_bytes());
            }
            clean::Vector(ref t) => write!(f.buf, "[{}]", **t),
            clean::FixedVector(ref t, ref s) => {
                write!(f.buf, "[{}, ..{}]", **t, *s);
            }
            clean::String => f.buf.write("str".as_bytes()),
            clean::Bool => f.buf.write("bool".as_bytes()),
            clean::Unit => f.buf.write("()".as_bytes()),
            clean::Bottom => f.buf.write("!".as_bytes()),
            clean::Unique(ref t) => write!(f.buf, "~{}", **t),
            clean::Managed(m, ref t) => {
                write!(f.buf, "@{}{}",
                       match m {
                           clean::Mutable => "mut ",
                           clean::Immutable => "",
                       }, **t)
            }
            clean::RawPointer(m, ref t) => {
                write!(f.buf, "*{}{}",
                       match m {
                           clean::Mutable => "mut ",
                           clean::Immutable => "",
                       }, **t)
            }
            clean::BorrowedRef{ lifetime: ref l, mutability, type_: ref ty} => {
                let lt = match *l { Some(ref l) => format!("{} ", *l), _ => ~"" };
                write!(f.buf, "&amp;{}{}{}",
                       lt,
                       match mutability {
                           clean::Mutable => "mut ",
                           clean::Immutable => "",
                       },
                       **ty);
            }
        }
    }
}

impl fmt::Default for clean::FnDecl {
    fn fmt(d: &clean::FnDecl, f: &mut fmt::Formatter) {
        let mut args = ~"";
        for (i, input) in d.inputs.iter().enumerate() {
            if i > 0 { args.push_str(", "); }
            if input.name.len() > 0 {
                args.push_str(format!("{}: ", input.name));
            }
            args.push_str(format!("{}", input.type_));
        }
        write!(f.buf, "({args}){arrow, select, yes{ -&gt; {ret}} other{}}",
               args = args,
               arrow = match d.output { clean::Unit => "no", _ => "yes" },
               ret = d.output);
    }
}

impl<'self> fmt::Default for Method<'self> {
    fn fmt(m: &Method<'self>, f: &mut fmt::Formatter) {
        let Method(selfty, d) = *m;
        let mut args = ~"";
        match *selfty {
            clean::SelfStatic => {},
            clean::SelfValue => args.push_str("self"),
            clean::SelfOwned => args.push_str("~self"),
            clean::SelfManaged(clean::Mutable) => args.push_str("@mut self"),
            clean::SelfManaged(clean::Immutable) => args.push_str("@self"),
            clean::SelfBorrowed(Some(ref lt), clean::Immutable) => {
                args.push_str(format!("&amp;{} self", *lt));
            }
            clean::SelfBorrowed(Some(ref lt), clean::Mutable) => {
                args.push_str(format!("&amp;{} mut self", *lt));
            }
            clean::SelfBorrowed(None, clean::Mutable) => {
                args.push_str("&amp;mut self");
            }
            clean::SelfBorrowed(None, clean::Immutable) => {
                args.push_str("&amp;self");
            }
        }
        for (i, input) in d.inputs.iter().enumerate() {
            if i > 0 || args.len() > 0 { args.push_str(", "); }
            if input.name.len() > 0 {
                args.push_str(format!("{}: ", input.name));
            }
            args.push_str(format!("{}", input.type_));
        }
        write!(f.buf, "({args}){arrow, select, yes{ -&gt; {ret}} other{}}",
               args = args,
               arrow = match d.output { clean::Unit => "no", _ => "yes" },
               ret = d.output);
    }
}

impl fmt::Default for VisSpace {
    fn fmt(v: &VisSpace, f: &mut fmt::Formatter) {
        match **v {
            Some(ast::public) => { write!(f.buf, "pub "); }
            Some(ast::private) => { write!(f.buf, "priv "); }
            Some(ast::inherited) | None => {}
        }
    }
}
