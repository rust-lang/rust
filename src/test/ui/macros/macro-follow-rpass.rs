// run-pass
#![allow(unused_macros)]
// Check the macro follow sets (see corresponding cfail test).

// FOLLOW(pat) = {FatArrow, Comma, Eq, Or, Ident(if), Ident(in)}
macro_rules! follow_pat {
    ($p:pat =>) => {};
    ($p:pat ,) => {};
    ($p:pat =) => {};
    ($p:pat |) => {};
    ($p:pat if) => {};
    ($p:pat in) => {};
}
// FOLLOW(expr) = {FatArrow, Comma, Semicolon}
macro_rules! follow_expr {
    ($e:expr =>) => {};
    ($e:expr ,) => {};
    ($e:expr ;) => {};
}
// FOLLOW(ty) = {OpenDelim(Brace), Comma, FatArrow, Colon, Eq, Gt, Semi, Or,
//               Ident(as), Ident(where), OpenDelim(Bracket), Nonterminal(Block)}
macro_rules! follow_ty {
    ($t:ty {}) => {};
    ($t:ty ,) => {};
    ($t:ty =>) => {};
    ($t:ty :) => {};
    ($t:ty =) => {};
    ($t:ty >) => {};
    ($t:ty ;) => {};
    ($t:ty |) => {};
    ($t:ty as) => {};
    ($t:ty where) => {};
    ($t:ty []) => {};
    ($t:ty $b:block) => {};
}
// FOLLOW(stmt) = FOLLOW(expr)
macro_rules! follow_stmt {
    ($s:stmt =>) => {};
    ($s:stmt ,) => {};
    ($s:stmt ;) => {};
}
// FOLLOW(path) = FOLLOW(ty)
macro_rules! follow_path {
    ($p:path {}) => {};
    ($p:path ,) => {};
    ($p:path =>) => {};
    ($p:path :) => {};
    ($p:path =) => {};
    ($p:path >) => {};
    ($p:path ;) => {};
    ($p:path |) => {};
    ($p:path as) => {};
    ($p:path where) => {};
    ($p:path []) => {};
    ($p:path $b:block) => {};
}
// FOLLOW(block) = any token
macro_rules! follow_block {
    ($b:block ()) => {};
    ($b:block []) => {};
    ($b:block {}) => {};
    ($b:block ,) => {};
    ($b:block =>) => {};
    ($b:block :) => {};
    ($b:block =) => {};
    ($b:block >) => {};
    ($b:block ;) => {};
    ($b:block |) => {};
    ($b:block +) => {};
    ($b:block ident) => {};
    ($b:block $p:pat) => {};
    ($b:block $e:expr) => {};
    ($b:block $t:ty) => {};
    ($b:block $s:stmt) => {};
    ($b:block $p:path) => {};
    ($b:block $c:block) => {};
    ($b:block $i:ident) => {};
    ($b:block $t:tt) => {};
    ($b:block $i:item) => {};
    ($b:block $m:meta) => {};
}
// FOLLOW(ident) = any token
macro_rules! follow_ident {
    ($i:ident ()) => {};
    ($i:ident []) => {};
    ($i:ident {}) => {};
    ($i:ident ,) => {};
    ($i:ident =>) => {};
    ($i:ident :) => {};
    ($i:ident =) => {};
    ($i:ident >) => {};
    ($i:ident ;) => {};
    ($i:ident |) => {};
    ($i:ident +) => {};
    ($i:ident ident) => {};
    ($i:ident $p:pat) => {};
    ($i:ident $e:expr) => {};
    ($i:ident $t:ty) => {};
    ($i:ident $s:stmt) => {};
    ($i:ident $p:path) => {};
    ($i:ident $b:block) => {};
    ($i:ident $j:ident) => {};
    ($i:ident $t:tt) => {};
    ($i:ident $j:item) => {};
    ($i:ident $m:meta) => {};
}
// FOLLOW(tt) = any token
macro_rules! follow_tt {
    ($t:tt ()) => {};
    ($t:tt []) => {};
    ($t:tt {}) => {};
    ($t:tt ,) => {};
    ($t:tt =>) => {};
    ($t:tt :) => {};
    ($t:tt =) => {};
    ($t:tt >) => {};
    ($t:tt ;) => {};
    ($t:tt |) => {};
    ($t:tt +) => {};
    ($t:tt ident) => {};
    ($t:tt $p:pat) => {};
    ($t:tt $e:expr) => {};
    ($t:tt $v:ty) => {};
    ($t:tt $s:stmt) => {};
    ($t:tt $p:path) => {};
    ($t:tt $b:block) => {};
    ($t:tt $i:ident) => {};
    ($t:tt $v:tt) => {};
    ($t:tt $i:item) => {};
    ($t:tt $m:meta) => {};
}
// FOLLOW(item) = any token
macro_rules! follow_item {
    ($i:item ()) => {};
    ($i:item []) => {};
    ($i:item {}) => {};
    ($i:item ,) => {};
    ($i:item =>) => {};
    ($i:item :) => {};
    ($i:item =) => {};
    ($i:item >) => {};
    ($i:item ;) => {};
    ($i:item |) => {};
    ($i:item +) => {};
    ($i:item ident) => {};
    ($i:item $p:pat) => {};
    ($i:item $e:expr) => {};
    ($i:item $t:ty) => {};
    ($i:item $s:stmt) => {};
    ($i:item $p:path) => {};
    ($i:item $b:block) => {};
    ($i:item $j:ident) => {};
    ($i:item $t:tt) => {};
    ($i:item $j:item) => {};
    ($i:item $m:meta) => {};
}
// FOLLOW(meta) = any token
macro_rules! follow_meta {
    ($m:meta ()) => {};
    ($m:meta []) => {};
    ($m:meta {}) => {};
    ($m:meta ,) => {};
    ($m:meta =>) => {};
    ($m:meta :) => {};
    ($m:meta =) => {};
    ($m:meta >) => {};
    ($m:meta ;) => {};
    ($m:meta |) => {};
    ($m:meta +) => {};
    ($m:meta ident) => {};
    ($m:meta $p:pat) => {};
    ($m:meta $e:expr) => {};
    ($m:meta $t:ty) => {};
    ($m:meta $s:stmt) => {};
    ($m:meta $p:path) => {};
    ($m:meta $b:block) => {};
    ($m:meta $i:ident) => {};
    ($m:meta $t:tt) => {};
    ($m:meta $i:item) => {};
    ($m:meta $n:meta) => {};
}

fn main() {}
