use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;

use rustc_hir::intravisit::{self, Visitor, FnKind};
use rustc_hir::{BodyId, Expr, ExprKind, FnDecl, HirId};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::Span;
use rustc_span::source_map::SourceMap;
//
use rustc_middle::ty::print::with_no_trimmed_paths;

#[derive(Clone, Debug)]
pub struct Found {
    pub span: Span,
    pub src: String,
    pub args: FxHashMap<String, String>,
}

pub fn search_pointcuts<'tcx>(tcx: TyCtxt<'tcx>, desc: String, src_map: &SourceMap) -> Vec<Found> {
    let pc = Pointcut::parse(&desc);
    pc.validate();
    println!("[AOP] pointcut accepted: {}, {:?}", desc, pc);

    let mut finder = FindCallExprs::new(tcx, pc, src_map);
    finder.prepare_cache();

    tcx.hir().visit_all_item_likes(&mut finder.as_deep_visitor());

    return finder.founds.clone();
}

struct FindCallExprs<'s, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub src_map: &'s SourceMap,
    pub pc: Pointcut,
    pub founds: Vec<Found>,
    pub name_id_cache: FxHashMap<String, DefId>,
}

impl<'s, 'tcx> intravisit::Visitor<'tcx> for FindCallExprs<'s, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_fn(
        &mut self,
        fk: FnKind<'tcx>,
        fd: &'tcx FnDecl<'tcx>,
        b: BodyId,
        s: Span,
        id: HirId,
    ) {
        match fk {
            FnKind::ItemFn(ident, ..) => {
                let fn_name = &ident.as_str();
                if self.pc.is_finding(fn_name) {
                    println!(
                        "[Aspect] Find {}, LocalDefId: {:?}, Ty: {:?}",
                        fn_name,
                        id.owner.local_def_index,
                        self.sema_ty(id)
                    );
                }
                if self.pc.is_enter_or_exit(fn_name) {
                    //
                    //
                }
            }
            FnKind::Method(ident, ..) => {
                let fn_name = &ident.as_str();
                if self.pc.is_finding(fn_name) {
                    println!(
                        "[Aspect] Find {}, LocalDefId: {:?}, Ty: {:?}",
                        fn_name,
                        id.owner.local_def_index,
                        self.sema_ty(id)
                    );
                }
                if self.pc.is_enter_or_exit(fn_name) {
                    //
                    //
                }
            }
            _ => {}
        }
        intravisit::walk_fn(self, fk, fd, b, s, id);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(_f, _args) = expr.kind {
            if let PDecl::Call {..} = self.pc.decl {
                if let Some(binds) = self.bind(&expr.kind) {
                    if self.check_conds(&binds) {
                        let found = Found {
                            span: expr.span,
                            src : self.expr_src(expr),
                            args: binds.iter().map(|(k, v)| (k.clone(), v.1.clone())).collect()
                        };
                        self.founds.push(found);
                    }
                }
            }
        }
        if let ExprKind::MethodCall(f, _span, _args, _argspan) = expr.kind {
            let name = f.ident.name.as_str();
            if self.pc.is_call(&name) {
                if let Some(binds) = self.bind(&expr.kind) {
                    if self.check_conds(&binds) {
                        let found = Found {
                            span: expr.span,
                            src: self.expr_src(expr),
                            args: binds.iter().map(|(k, v)| (k.clone(), v.1.clone())).collect()
                        };
                        self.founds.push(found);
                    }
                }
            }
        }
        intravisit::walk_expr(self, expr);
    }
}

impl<'s, 'tcx> FindCallExprs<'s, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, pc: Pointcut, src_map: &'s SourceMap) -> Self {
        FindCallExprs { tcx, src_map, pc, founds: Vec::new(), name_id_cache: FxHashMap::default() }
    }

    fn expr_src(&self, e: &'tcx Expr<'tcx>) -> String {
        self.src_map.span_to_snippet(e.span).unwrap_or(String::new())
    }

    fn prepare_cache(&mut self) {
        for &did in self.tcx.all_traits(()) {
            let name = self
                .tcx
                .opt_item_name(did)
                .map(|n| n.as_str().to_string())
                .unwrap_or(String::new());
            self.name_id_cache.insert(name, did);
        }
    }

    fn check_conds(&self, binds: &FxHashMap<String, (HirId, String)>) -> bool {
        for c in &self.pc.conditions {
            match c {
                Constraint::Type { name, ty_name } => {
                    if let Some((hir_id, src_str)) = binds.get(name) {
                        let real_ty = self.sema_ty(*hir_id);
                        let real_ty = with_no_trimmed_paths(|| real_ty.to_string());
                        println!("[DEBUG] name {} str {} real_ty {} cond_ty {}", name, src_str, real_ty, ty_name);
                        if ty_name.trim().ends_with('*') {
                            let ty_name_prefix = ty_name.trim().trim_matches('*');
                            if real_ty.starts_with(ty_name_prefix) {
                                continue;
                            }
                        } else if real_ty == *ty_name {
                            continue;
                        } else {
                            println!("[DEBUG] mismatch name {} hir_id {:?} type {}", name, hir_id, real_ty);
                        }
                    }
                    return false;
                }
                Constraint::ImplTrait {..} => {

                }
            }
        }
        return true;
    }

    fn sema_ty(&self, hir_id: HirId) -> Ty<'tcx> {
        let parent_id = self.tcx.hir().get_parent_item(hir_id);
        let body = self.tcx.hir().body_owned_by(parent_id);
        let ty_res = self.tcx.typeck_body(body);
        ty_res.node_type_opt(hir_id).unwrap_or(self.tcx.mk_unit())
    }

    fn bind(&self, expr: &'tcx ExprKind<'tcx>) -> Option<FxHashMap<String, (HirId, String)>> {
        let mut vars = FxHashMap::default();
        match &self.pc.decl {
            PDecl::Call { ref fn_name, ref args } => {
                let pc_args = args;
                if let ExprKind::Call(e, target_args) = expr {
                    let actual_name = self.expr_src(e);
                    if !fn_name.starts_with('_') && fn_name != &actual_name {
                        return None;
                    }
                    if fn_name.starts_with('_') {
                        vars.insert(fn_name.to_string(), (e.hir_id, actual_name));
                    }
                    if let Some(pc_args) = pc_args {
                        if pc_args.contains(&"*".to_string()) {
                            return Some(vars)
                        }
                        if pc_args.len() != target_args.len() {
                            return None
                        }
                        for i in 0..pc_args.len() {
                            let pc_arg = pc_args[i].to_string();
                            let target_arg = self.expr_src(&target_args[i]);
                            if pc_arg.starts_with('_') {
                                vars.insert(pc_arg, (target_args[i].hir_id, target_arg));
                            } else if pc_arg == "*" {
                                continue;
                            } else {
                                if pc_arg != target_arg {
                                    return None;
                                }
                            }
                        }
                    }
                } else {
                    return None
                }
            }
            PDecl::Method { ref receiver_name, ref fn_name, ref args }  => {
                let pc_args = args;
                if let ExprKind::MethodCall(e, _, target_args, _) = expr {
                    let actual_name = e.ident.as_str().to_string();
                    if !fn_name.starts_with('_') && fn_name != &actual_name {
                        return None;
                    }
                    if fn_name.starts_with('_') && e.hir_id.is_some() {
                        vars.insert(fn_name.to_string(), (e.hir_id.unwrap(), actual_name));
                    }
                    let actual_receiver_name = self.expr_src(&target_args[0]);
                    if !receiver_name.starts_with('_') && actual_receiver_name != *receiver_name {
                        return None
                    }
                    if receiver_name.starts_with('_') && e.hir_id.is_some() {
                        vars.insert(receiver_name.to_string(), (target_args[0].hir_id, actual_receiver_name));
                    }
                    if let Some(pc_args) = pc_args {
                        if pc_args.contains(&"*".to_string()) {
                            return Some(vars)
                        }
                        if pc_args.len() + 1 != target_args.len() {
                            return None
                        }
                        for i in 0..pc_args.len() {
                            let pc_arg = pc_args[i].to_string();
                            let target_arg = self.expr_src(&target_args[i]);
                            if pc_arg.starts_with('_') {
                                vars.insert(pc_arg, (target_args[i+1].hir_id, target_arg));
                            } else if pc_arg == "*" {
                                continue;
                            } else {
                                if pc_arg != target_arg {
                                    return None;
                                }
                            }
                        }
                    }
                } else {
                    return None
                }
            }
            _ => {}
        }
        Some(vars)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Pointcut {
    decl: PDecl,
    conditions: Vec<Constraint>,
}

#[derive(Debug, Clone, PartialEq)]
enum PDecl {
    Call { fn_name: String, args: Option<Vec<String>> },
    Method { receiver_name: String, fn_name: String, args: Option<Vec<String>> },
    Enter { path: String },
    Exit { path: String },
    Find { ident: String },
}

#[derive(Debug, Clone, PartialEq)]
enum Constraint {
    Type { name: String, ty_name: String },
    ImplTrait { name: String, trait_name: String },
}

impl Pointcut {
    fn parse(s: &str) -> Self {
        let mut p = PointcutParser { tokens: lex(s), cur: 0};
        //println!("[DEBUG] tokens: {:?}", p.tokens);
        p.parse_pointcut()
    }

    fn is_enter_or_exit(&self, s: &str) -> bool {
        match &self.decl {
            PDecl::Enter { path } => path == s,
            PDecl::Exit { path } => path == s,
            _ => false,
        }
    }

    fn is_finding(&self, s: &str) -> bool {
        match &self.decl {
            PDecl::Find { ident } => ident == s,
            _ => false
        }
    }

    fn is_call(&self, s: &str) -> bool {
        match &self.decl {
            PDecl::Call { fn_name, .. } => fn_name == s,
            PDecl::Method { fn_name, .. } =>  fn_name == s,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum PointcutToken {
    Dot,     // .
    LParen,  // (
    RParen,  // )
    Star,    // *
    LAngle,  // <
    RAngle,  // >
    Colon,   // :
    ColonColon, // ::
    AndAnd,     // &&
    Comma,      // ,
    Name(String),
}

impl PointcutToken {
    fn is_name(&self, n: &str) -> bool {
        if let PointcutToken::Name(ref name) = self {
            n == name
        } else {
            false
        }
    }
}

impl std::fmt::Display for PointcutToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use PointcutToken::*;
        match self {
            Dot => write!(f, "{}", "."),
            LParen => write!(f, "{}", "("),
            RParen => write!(f, "{}", ")"),
            Star => write!(f, "{}", "*"),
            LAngle => write!(f, "{}", "<"),
            RAngle => write!(f, "{}", ">"),
            Colon => write!(f, "{}", ":"),
            ColonColon => write!(f, "{}", "::"),
            AndAnd => write!(f, "{}", "&&"),
            Comma => write!(f, "{}", ","),
            Name(s) => write!(f, "{}", s),
        }
    }
}

fn collect(vars: &mut FxHashMap<String, bool>, v: &str) {
    if v.len() > 1 && v.starts_with('_') {
        vars.insert(v.to_string(), false);
    }
}

impl Pointcut {
    fn collect_vars(&self) -> FxHashMap<String, bool> {
        let mut vars = FxHashMap::default();
        match &self.decl {
            PDecl::Call { ref fn_name, ref args } => {
                collect(&mut vars, fn_name);
                for inside in args {
                    for arg in inside {
                        collect(&mut vars, arg);
                    }
                }
            }
            PDecl::Method { ref receiver_name, ref fn_name, ref args } => {
                collect(&mut vars, receiver_name);
                collect(&mut vars, fn_name);
                for inside in args {
                    for arg in inside {
                        collect(&mut vars, arg);
                    }
                }
            }
            _ => {}
        }
        vars
    }

    fn validate(&self) {
        let vars = self.collect_vars();
        for c in &self.conditions {
            let name = match c {
                Constraint::Type { name, .. } => name,
                Constraint::ImplTrait { .. } => {
                    fast_err("impl trait condition is not implemented yet")
                }
            };

            if !vars.contains_key(name) {
                fast_err(format!("Variable '{}' is not defined", name))
            }
        }
    }
}

fn fast_err(msg: impl ToString) -> ! {
    eprintln!("Parse input pointcut error: {}", msg.to_string());
    std::process::exit(1)
}

fn lex(src: &str) -> Vec<PointcutToken> {
    let mut cur: usize = 0;
    let bytes = src.as_bytes();
    let mut tokens = Vec::<PointcutToken>::new();
    while cur < bytes.len() {
        match bytes[cur] {
            b' ' => {}
            b'.' => { tokens.push(PointcutToken::Dot); }
            b'(' => { tokens.push(PointcutToken::LParen); }
            b')' => { tokens.push(PointcutToken::RParen); }
            b'*' => { tokens.push(PointcutToken::Star); }
            b'<' => { tokens.push(PointcutToken::LAngle); }
            b'>' => { tokens.push(PointcutToken::RAngle); }
            b',' => { tokens.push(PointcutToken::Comma); }
            b':' => {
                if cur + 1 < bytes.len() && bytes[cur+1] == b':' {
                    cur += 1;
                    tokens.push(PointcutToken::ColonColon);
                } else {
                    tokens.push(PointcutToken::Colon);
                }
            }
            b'&' if bytes.get(cur+1) == Some(&b'&') => {
                cur +=1 ;
                tokens.push(PointcutToken::AndAnd);
            }
            _ => {
                let b = bytes[cur];
                let is_alpha_num = |b: u8| {
                    b == b'_' || b == b'&'
                    || (b >= b'a' && b <= b'z')
                    || (b >= b'A' && b <= b'Z')
                    || (b >= b'0' && b <= b'9')
                };
                let mut end = cur;
                while end < bytes.len() && is_alpha_num(bytes[end]) {
                    end += 1;
                }
                if end > cur {
                    let ident = src[cur..end].to_string();
                    tokens.push(PointcutToken::Name(ident));
                    cur = end;
                    continue;
                } else {
                    fast_err(format!("Unknown token {:?}.", char::from(b)))
                }
            }
        }
        cur += 1;
    }
    return tokens;
}

struct PointcutParser {
    tokens: Vec<PointcutToken>,
    cur: usize,
}

impl PointcutParser {
    fn peek(&self) -> Option<PointcutToken> {
        self.tokens.get(self.cur).cloned()
    }

    fn next(&mut self) -> PointcutToken {
        if let Some(t) = self.peek() {
            self.cur += 1;
            return t;
        } else {
            fast_err("Expected a token, bot nothing remained.")
        }
    }

    fn expect_token(&mut self, t: PointcutToken) {
        if self.next() != t {
            fast_err(format!("Expected token {}, but not found.", t))
        }
    }

    fn expect_name(&mut self) -> String {
        let n = self.next();
        if let PointcutToken::Name(t) = n {
            return t;
        } else {
            fast_err(format!("Expect identifier, but '{}' is found", n))
        }
    }

    fn parse_pointcut(&mut self) -> Pointcut {
        let decl = self.parse_decl();
        let mut pc = Pointcut { decl, conditions: Vec::new() };
        if let Some(_) = self.peek() {
            let n = self.expect_name();
            if n != "where" {
                fast_err(format!("Expect 'where', but '{}' is found.", n))
            }
        } else {
            return pc;
        }

        let cond = self.parse_cond();
        pc.conditions.push(cond);
        while self.peek() == Some(PointcutToken::AndAnd) {
            self.cur += 1;
            let cond = self.parse_cond();
            pc.conditions.push(cond);
        }

        if let Some(t) = self.peek() {
            fast_err(format!("Constraint ended with unknown token '{:?}'.", t))
        }
        return pc;
    }

    fn parse_decl(&mut self) -> PDecl {
        let head = self.next();
        if head.is_name("enter") {
            let path = self.parse_path(); // TODO: add arguments
            return PDecl::Enter { path };
        } else if head.is_name("exit") {
            let path = self.parse_path();
            return PDecl::Exit { path };
        } else if head.is_name("find") {
            let ident = self.expect_name();
            return PDecl::Find { ident };
        } else if head.is_name("call") {
            let n = self.next();
            if let PointcutToken::Name(ident) = n {
                let p = self.next();
                if p == PointcutToken::Dot {
                    let name = self.expect_name();
                    self.expect_token(PointcutToken::LParen);
                    return PDecl::Method {
                        receiver_name: ident,
                        fn_name: name,
                        args: self.parse_args(),
                    };
                } else if p == PointcutToken::LParen {
                    return PDecl::Call { fn_name: ident, args: self.parse_args() };
                } else {
                    fast_err(format!("Unknown token '{:?}' is found", p))
                }
            } else {
                fast_err(format!("Expect identifier, but '{}' is found.", n))
            }
        } else {
            fast_err("Unknown command")
        }
    }

    fn parse_args(&mut self) -> Option<Vec<String>> {
        let arg = self.next();
        if arg == PointcutToken::Star {
            self.expect_token(PointcutToken::RParen);
            return None;
        }
        if arg == PointcutToken::RParen {
            return Some(Vec::new());
        }
        let mut args = Vec::new();
        if let PointcutToken::Name(arg_name) = arg {
            args.push(arg_name);
        } else {
            fast_err(format!("Expect argument, bug '{:?}' is found.", arg))
        }
        while self.peek() == Some(PointcutToken::Comma) {
            self.next();
            args.push(self.expect_name());
        }
        self.expect_token(PointcutToken::RParen);
        return Some(args);
    }

    fn parse_cond(&mut self) -> Constraint {
        let name = self.expect_name();
        if !name.starts_with('_') {
            fast_err(format!("Only variable can be used in condition, but '{}' is found.", name))
        }
        let punc = self.next();
        if punc == PointcutToken::Colon {
            let ty_name = self.parse_path();
            return Constraint::Type { name, ty_name }
        } else if punc.is_name("impl") {
            let trait_name = self.parse_path();
            return Constraint::ImplTrait { name, trait_name }
        } else {
            fast_err(format!("Expect colon or impl, but '{:?}' is found.", punc))
        }
    }

    fn parse_path(&mut self) -> String {
        let is_seg = | t: &Option<PointcutToken> | {
            use PointcutToken::*;
            match t {
                &Some(Name(ref n)) if !n.starts_with('_') => true,
                &Some(LAngle) | &Some(RAngle) | &Some(ColonColon) | &Some(Star) => true,
                _ => false,
            }
        };
        let mut r = String::new();
        let mut next = self.peek();
        while is_seg(&next) {
            let n = self.next();
            r.push_str(&n.to_string());
            next = self.peek();
        }
        return r;
    }
}
