use std::collections::BTreeSet;
use std::path::Path;

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Expr, ExprCall, ExprForLoop, ExprLoop, ExprMacro, ExprMethodCall, ExprPath, ExprWhile, Local,
    Macro, Pat,
};

use crate::model::{DiffClass, RuleHit};

pub fn scan_file(path: &Path, is_build_script: bool, is_proc_macro: bool) -> Vec<RuleHit> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let path_str = path.to_string_lossy().into_owned();
    let has_out_dir_ref = text.contains("OUT_DIR");

    let Ok(file) = syn::parse_file(&text) else {
        return fallback_scan(&text, &path_str, is_build_script, is_proc_macro, has_out_dir_ref);
    };

    let mut visitor =
        SynScanVisitor::new(path_str, is_build_script, is_proc_macro, has_out_dir_ref);
    visitor.visit_file(&file);
    visitor.finish()
}

struct SynScanVisitor {
    path: String,
    is_build_script: bool,
    is_proc_macro: bool,
    has_out_dir_ref: bool,
    out_dir_bindings: BTreeSet<String>,
    loop_depth: usize,
    hits: Vec<RuleHit>,
    seen: BTreeSet<(String, usize, String)>,
}

impl SynScanVisitor {
    fn new(
        path: String,
        is_build_script: bool,
        is_proc_macro: bool,
        has_out_dir_ref: bool,
    ) -> Self {
        Self {
            path,
            is_build_script,
            is_proc_macro,
            has_out_dir_ref,
            out_dir_bindings: BTreeSet::new(),
            loop_depth: 0,
            hits: Vec::new(),
            seen: BTreeSet::new(),
        }
    }

    fn finish(self) -> Vec<RuleHit> {
        self.hits
    }

    fn push_hit(
        &mut self,
        rule_id: &str,
        class_hint: DiffClass,
        line: usize,
        strength: &str,
        detail: &str,
        fix_hint: &str,
    ) {
        let key = (rule_id.to_string(), line, detail.to_string());
        if !self.seen.insert(key) {
            return;
        }

        self.hits.push(RuleHit {
            rule_id: rule_id.to_string(),
            class_hint,
            path: self.path.clone(),
            line,
            strength: strength.to_string(),
            detail: detail.to_string(),
            fix_hint: fix_hint.to_string(),
        });
    }

    fn push_proc_macro_side_effect(&mut self, line: usize, detail: &str) {
        if !self.is_proc_macro {
            return;
        }
        self.push_hit(
            "RE007",
            DiffClass::ProcMacro,
            line,
            "high",
            detail,
            "avoid ambient side effects in proc-macro expansion",
        );
    }

    fn check_call_path(&mut self, path: &syn::Path, span: Span) {
        let line = line_of_span(span);
        if ends_with(path, &["read_dir"]) {
            let strength =
                if self.is_build_script || self.is_proc_macro { "high" } else { "medium" };
            self.push_hit(
                "RE001",
                DiffClass::UnstableOrder,
                line,
                strength,
                "unsorted read_dir usage",
                "collect entries then sort before emitting",
            );
        }

        if ends_with(path, &["SystemTime", "now"]) || ends_with(path, &["Instant", "now"]) {
            self.push_hit(
                "RE003",
                DiffClass::Timestamp,
                line,
                "high",
                "wall-clock value influences output",
                "prefer SOURCE_DATE_EPOCH or avoid embedding timestamps",
            );
            self.push_proc_macro_side_effect(line, "proc-macro uses wall clock");
        }

        if ends_with(path, &["env", "var"])
            || ends_with(path, &["env", "vars"])
            || ends_with(path, &["env", "var_os"])
            || ends_with(path, &["env", "vars_os"])
            || ends_with(path, &["env", "current_dir"])
            || ends_with(path, &["current_dir"])
        {
            self.push_hit(
                "RE004",
                DiffClass::EnvLeak,
                line,
                "medium",
                "ambient env/cwd usage detected",
                "make inputs explicit and deterministic",
            );
            self.push_proc_macro_side_effect(line, "proc-macro reads environment or cwd");
        }

        if ends_with(path, &["thread", "spawn"]) || ends_with(path, &["spawn"]) {
            self.push_hit(
                "RE008",
                DiffClass::ScheduleSensitiveParallelism,
                line,
                "medium",
                "parallel thread spawn pattern detected",
                "enforce deterministic reduce/collect ordering",
            );
        }
    }

    fn record_iteration_method(&mut self, method: &str, span: Span) {
        if matches!(method, "iter" | "into_iter" | "keys" | "values") {
            let line = line_of_span(span);
            let strength =
                if self.is_build_script || self.is_proc_macro { "high" } else { "medium" };
            self.push_hit(
                "RE002",
                DiffClass::UnstableOrder,
                line,
                strength,
                "potential HashMap/HashSet iteration to sink",
                "use deterministic order (BTreeMap/BTreeSet or sort Vec)",
            );
        }
    }

    fn record_parallel_method(&mut self, method: &str, span: Span) {
        if matches!(method, "par_iter" | "into_par_iter" | "par_bridge" | "reduce" | "min" | "max")
        {
            self.push_hit(
                "RE008",
                DiffClass::ScheduleSensitiveParallelism,
                line_of_span(span),
                "medium",
                "parallel iterator/reduction pattern detected",
                "enforce deterministic reduce/collect ordering",
            );
        }
    }

    fn record_write(&mut self, line: usize, derived_from_out_dir: bool, explicit_path: bool) {
        if self.is_build_script && !derived_from_out_dir {
            let strength = if explicit_path || !self.has_out_dir_ref { "high" } else { "medium" };
            self.push_hit(
                "RE006",
                DiffClass::BuildScript,
                line,
                strength,
                "build script write may escape OUT_DIR",
                "for build scripts, write only under OUT_DIR",
            );
        }
        self.push_proc_macro_side_effect(line, "proc-macro performs filesystem write");
    }

    fn inspect_macro_expr(&mut self, expr: &ExprMacro) {
        self.inspect_macro(&expr.mac, expr.span());
    }

    fn inspect_macro(&mut self, mac: &Macro, span: Span) {
        let macro_name = mac.path.segments.last().map(|s| s.ident.to_string());
        let line = line_of_span(span);
        let tokens = mac.tokens.to_string().replace(' ', "");

        if macro_name.as_deref() == Some("println") && tokens.contains("cargo::") {
            let strength = if self.is_build_script && self.loop_depth > 0 {
                "high"
            } else if self.is_build_script {
                "medium"
            } else {
                "low"
            };
            self.push_hit(
                "RE005",
                DiffClass::BuildScript,
                line,
                strength,
                "cargo:: instruction emission detected",
                "if emitted in loop, pre-sort values before println!",
            );
        }

        if matches!(macro_name.as_deref(), Some("write") | Some("writeln")) {
            self.record_write(line, self.has_out_dir_ref, false);
        }
    }

    fn expr_is_out_dir_source(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Call(call) => {
                if let Expr::Path(ExprPath { path, .. }) = &*call.func {
                    if (ends_with(path, &["env", "var"]) || ends_with(path, &["env", "var_os"]))
                        && call.args.first().is_some_and(is_out_dir_literal)
                    {
                        return true;
                    }
                }
                self.expr_is_out_dir_source(&call.func)
                    || call.args.iter().any(|arg| self.expr_is_out_dir_source(arg))
            }
            Expr::MethodCall(method) => {
                self.expr_is_out_dir_source(&method.receiver)
                    || method.args.iter().any(|arg| self.expr_is_out_dir_source(arg))
            }
            Expr::Path(ExprPath { path, .. }) => path
                .segments
                .last()
                .is_some_and(|seg| self.out_dir_bindings.contains(&seg.ident.to_string())),
            Expr::Paren(expr) => self.expr_is_out_dir_source(&expr.expr),
            Expr::Try(expr) => self.expr_is_out_dir_source(&expr.expr),
            Expr::Reference(expr) => self.expr_is_out_dir_source(&expr.expr),
            _ => false,
        }
    }

    fn expr_mentions_out_dir(&self, expr: &Expr) -> bool {
        if self.expr_is_out_dir_source(expr) {
            return true;
        }

        match expr {
            Expr::Path(ExprPath { path, .. }) => path.segments.last().is_some_and(|seg| {
                let ident = seg.ident.to_string();
                ident == "OUT_DIR" || ident == "out_dir" || self.out_dir_bindings.contains(&ident)
            }),
            Expr::Lit(lit) => match &lit.lit {
                syn::Lit::Str(s) => s.value().contains("OUT_DIR"),
                _ => false,
            },
            Expr::Call(call) => {
                self.expr_mentions_out_dir(&call.func)
                    || call.args.iter().any(|arg| self.expr_mentions_out_dir(arg))
            }
            Expr::MethodCall(method) => {
                self.expr_mentions_out_dir(&method.receiver)
                    || method.args.iter().any(|arg| self.expr_mentions_out_dir(arg))
            }
            Expr::Macro(mac) => {
                let tokens = mac.mac.tokens.to_string();
                tokens.contains("OUT_DIR")
                    || tokens.contains("out_dir")
                    || self.out_dir_bindings.iter().any(|name| tokens.contains(name))
            }
            Expr::Binary(bin) => {
                self.expr_mentions_out_dir(&bin.left) || self.expr_mentions_out_dir(&bin.right)
            }
            Expr::Unary(expr) => self.expr_mentions_out_dir(&expr.expr),
            Expr::Paren(expr) => self.expr_mentions_out_dir(&expr.expr),
            Expr::Reference(expr) => self.expr_mentions_out_dir(&expr.expr),
            Expr::Try(expr) => self.expr_mentions_out_dir(&expr.expr),
            Expr::Await(expr) => self.expr_mentions_out_dir(&expr.base),
            Expr::Cast(expr) => self.expr_mentions_out_dir(&expr.expr),
            Expr::Index(expr) => {
                self.expr_mentions_out_dir(&expr.expr) || self.expr_mentions_out_dir(&expr.index)
            }
            Expr::Field(expr) => self.expr_mentions_out_dir(&expr.base),
            Expr::Tuple(expr) => expr.elems.iter().any(|e| self.expr_mentions_out_dir(e)),
            Expr::Array(expr) => expr.elems.iter().any(|e| self.expr_mentions_out_dir(e)),
            Expr::Struct(expr) => {
                expr.fields.iter().any(|f| self.expr_mentions_out_dir(&f.expr))
                    || expr.rest.as_deref().is_some_and(|rest| self.expr_mentions_out_dir(rest))
            }
            _ => false,
        }
    }
}

impl<'ast> Visit<'ast> for SynScanVisitor {
    fn visit_local(&mut self, node: &'ast Local) {
        if let Pat::Ident(pat_ident) = &node.pat {
            if let Some(init) = &node.init {
                if self.expr_is_out_dir_source(&init.expr) {
                    self.out_dir_bindings.insert(pat_ident.ident.to_string());
                }
            }
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(ExprPath { path, .. }) = &*node.func {
            self.check_call_path(path, node.span());
            if is_write_call_path(path) {
                let derived_from_out_dir =
                    node.args.first().is_some_and(|arg| self.expr_mentions_out_dir(arg));
                self.record_write(line_of_span(node.span()), derived_from_out_dir, true);
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        self.record_iteration_method(&method, node.span());
        self.record_parallel_method(&method, node.span());
        if method == "write_all" {
            self.record_write(line_of_span(node.span()), self.has_out_dir_ref, false);
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_macro(&mut self, node: &'ast ExprMacro) {
        self.inspect_macro_expr(node);
        visit::visit_expr_macro(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        self.inspect_macro(node, node.span());
        visit::visit_macro(self, node);
    }

    fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
        self.loop_depth += 1;
        visit::visit_expr_for_loop(self, node);
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    fn visit_expr_while(&mut self, node: &'ast ExprWhile) {
        self.loop_depth += 1;
        visit::visit_expr_while(self, node);
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    fn visit_expr_loop(&mut self, node: &'ast ExprLoop) {
        self.loop_depth += 1;
        visit::visit_expr_loop(self, node);
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }
}

fn line_of_span(span: Span) -> usize {
    span.start().line.max(1)
}

fn ends_with(path: &syn::Path, suffix: &[&str]) -> bool {
    let segments = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>();
    if segments.len() < suffix.len() {
        return false;
    }
    segments[segments.len() - suffix.len()..].iter().zip(suffix.iter()).all(|(a, b)| a == b)
}

fn is_write_call_path(path: &syn::Path) -> bool {
    ends_with(path, &["fs", "write"]) || ends_with(path, &["File", "create"])
}

fn is_out_dir_literal(expr: &Expr) -> bool {
    if let Expr::Lit(lit) = expr {
        if let syn::Lit::Str(s) = &lit.lit {
            return s.value() == "OUT_DIR";
        }
    }
    false
}

fn fallback_scan(
    text: &str,
    path: &str,
    is_build_script: bool,
    is_proc_macro: bool,
    has_out_dir_ref: bool,
) -> Vec<RuleHit> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        if line.contains("read_dir(") {
            out.push(RuleHit {
                rule_id: "RE001".to_string(),
                class_hint: DiffClass::UnstableOrder,
                path: path.to_string(),
                line: line_no,
                strength: if is_build_script || is_proc_macro { "high" } else { "medium" }
                    .to_string(),
                detail: "unsorted read_dir usage (fallback)".to_string(),
                fix_hint: "collect entries then sort before emitting".to_string(),
            });
        }
        if line.contains("println!") && line.contains("cargo::") {
            out.push(RuleHit {
                rule_id: "RE005".to_string(),
                class_hint: DiffClass::BuildScript,
                path: path.to_string(),
                line: line_no,
                strength: if is_build_script { "medium" } else { "low" }.to_string(),
                detail: "cargo:: instruction emission detected (fallback)".to_string(),
                fix_hint: "if emitted in loop, pre-sort values before println!".to_string(),
            });
        }
        if line.contains("SystemTime::now") || line.contains("Instant::now") {
            out.push(RuleHit {
                rule_id: "RE003".to_string(),
                class_hint: DiffClass::Timestamp,
                path: path.to_string(),
                line: line_no,
                strength: "high".to_string(),
                detail: "wall-clock value influences output (fallback)".to_string(),
                fix_hint: "prefer SOURCE_DATE_EPOCH or avoid embedding timestamps".to_string(),
            });
        }
        if (line.contains("fs::write")
            || line.contains("File::create")
            || line.contains("write_all"))
            && is_build_script
            && !line.contains("OUT_DIR")
            && !has_out_dir_ref
        {
            out.push(RuleHit {
                rule_id: "RE006".to_string(),
                class_hint: DiffClass::BuildScript,
                path: path.to_string(),
                line: line_no,
                strength: if !has_out_dir_ref { "high" } else { "medium" }.to_string(),
                detail: "file write call detected (fallback)".to_string(),
                fix_hint: "for build scripts, write only under OUT_DIR".to_string(),
            });
        }
    }
    out
}

#[cfg(test)]
#[path = "tests/syn_scan.rs"]
mod tests;
