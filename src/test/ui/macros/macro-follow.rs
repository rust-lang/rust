//
// Check the macro follow sets (see corresponding rpass test).

#![allow(unused_macros)]

// FOLLOW(pat) = {FatArrow, Comma, Eq, Or, Ident(if), Ident(in)}
macro_rules! follow_pat {
    ($p:pat ()) => {};       //~ERROR  `$p:pat` is followed by `(`
    ($p:pat []) => {};       //~ERROR  `$p:pat` is followed by `[`
    ($p:pat {}) => {};       //~ERROR  `$p:pat` is followed by `{`
    ($p:pat :) => {};        //~ERROR `$p:pat` is followed by `:`
    ($p:pat >) => {};        //~ERROR `$p:pat` is followed by `>`
    ($p:pat +) => {};        //~ERROR `$p:pat` is followed by `+`
    ($p:pat ident) => {};    //~ERROR `$p:pat` is followed by `ident`
    ($p:pat $q:pat) => {};   //~ERROR `$p:pat` is followed by `$q:pat`
    ($p:pat $e:expr) => {};  //~ERROR `$p:pat` is followed by `$e:expr`
    ($p:pat $t:ty) => {};    //~ERROR `$p:pat` is followed by `$t:ty`
    ($p:pat $s:stmt) => {};  //~ERROR `$p:pat` is followed by `$s:stmt`
    ($p:pat $q:path) => {};  //~ERROR `$p:pat` is followed by `$q:path`
    ($p:pat $b:block) => {}; //~ERROR `$p:pat` is followed by `$b:block`
    ($p:pat $i:ident) => {}; //~ERROR `$p:pat` is followed by `$i:ident`
    ($p:pat $t:tt) => {};    //~ERROR `$p:pat` is followed by `$t:tt`
    ($p:pat $i:item) => {};  //~ERROR `$p:pat` is followed by `$i:item`
    ($p:pat $m:meta) => {};  //~ERROR `$p:pat` is followed by `$m:meta`
}
// FOLLOW(expr) = {FatArrow, Comma, Semicolon}
macro_rules! follow_expr {
    ($e:expr ()) => {};       //~ERROR  `$e:expr` is followed by `(`
    ($e:expr []) => {};       //~ERROR  `$e:expr` is followed by `[`
    ($e:expr {}) => {};       //~ERROR  `$e:expr` is followed by `{`
    ($e:expr =) => {};        //~ERROR `$e:expr` is followed by `=`
    ($e:expr |) => {};        //~ERROR `$e:expr` is followed by `|`
    ($e:expr :) => {};        //~ERROR `$e:expr` is followed by `:`
    ($e:expr >) => {};        //~ERROR `$e:expr` is followed by `>`
    ($e:expr +) => {};        //~ERROR `$e:expr` is followed by `+`
    ($e:expr ident) => {};    //~ERROR `$e:expr` is followed by `ident`
    ($e:expr if) => {};       //~ERROR `$e:expr` is followed by `if`
    ($e:expr in) => {};       //~ERROR `$e:expr` is followed by `in`
    ($e:expr $p:pat) => {};   //~ERROR `$e:expr` is followed by `$p:pat`
    ($e:expr $f:expr) => {};  //~ERROR `$e:expr` is followed by `$f:expr`
    ($e:expr $t:ty) => {};    //~ERROR `$e:expr` is followed by `$t:ty`
    ($e:expr $s:stmt) => {};  //~ERROR `$e:expr` is followed by `$s:stmt`
    ($e:expr $p:path) => {};  //~ERROR `$e:expr` is followed by `$p:path`
    ($e:expr $b:block) => {}; //~ERROR `$e:expr` is followed by `$b:block`
    ($e:expr $i:ident) => {}; //~ERROR `$e:expr` is followed by `$i:ident`
    ($e:expr $t:tt) => {};    //~ERROR `$e:expr` is followed by `$t:tt`
    ($e:expr $i:item) => {};  //~ERROR `$e:expr` is followed by `$i:item`
    ($e:expr $m:meta) => {};  //~ERROR `$e:expr` is followed by `$m:meta`
}
// FOLLOW(ty) = {OpenDelim(Brace), Comma, FatArrow, Colon, Eq, Gt, Semi, Or,
//               Ident(as), Ident(where), OpenDelim(Bracket), Nonterminal(Block)}
macro_rules! follow_ty {
    ($t:ty ()) => {};       //~ERROR  `$t:ty` is followed by `(`
    ($t:ty []) => {};       // ok (RFC 1462)
    ($t:ty +) => {};        //~ERROR `$t:ty` is followed by `+`
    ($t:ty ident) => {};    //~ERROR `$t:ty` is followed by `ident`
    ($t:ty if) => {};       //~ERROR `$t:ty` is followed by `if`
    ($t:ty $p:pat) => {};   //~ERROR `$t:ty` is followed by `$p:pat`
    ($t:ty $e:expr) => {};  //~ERROR `$t:ty` is followed by `$e:expr`
    ($t:ty $r:ty) => {};    //~ERROR `$t:ty` is followed by `$r:ty`
    ($t:ty $s:stmt) => {};  //~ERROR `$t:ty` is followed by `$s:stmt`
    ($t:ty $p:path) => {};  //~ERROR `$t:ty` is followed by `$p:path`
    ($t:ty $b:block) => {}; // ok (RFC 1494)
    ($t:ty $i:ident) => {}; //~ERROR `$t:ty` is followed by `$i:ident`
    ($t:ty $r:tt) => {};    //~ERROR `$t:ty` is followed by `$r:tt`
    ($t:ty $i:item) => {};  //~ERROR `$t:ty` is followed by `$i:item`
    ($t:ty $m:meta) => {};  //~ERROR `$t:ty` is followed by `$m:meta`
}
// FOLLOW(stmt) = FOLLOW(expr)
macro_rules! follow_stmt {
    ($s:stmt ()) => {};       //~ERROR  `$s:stmt` is followed by `(`
    ($s:stmt []) => {};       //~ERROR  `$s:stmt` is followed by `[`
    ($s:stmt {}) => {};       //~ERROR  `$s:stmt` is followed by `{`
    ($s:stmt =) => {};        //~ERROR `$s:stmt` is followed by `=`
    ($s:stmt |) => {};        //~ERROR `$s:stmt` is followed by `|`
    ($s:stmt :) => {};        //~ERROR `$s:stmt` is followed by `:`
    ($s:stmt >) => {};        //~ERROR `$s:stmt` is followed by `>`
    ($s:stmt +) => {};        //~ERROR `$s:stmt` is followed by `+`
    ($s:stmt ident) => {};    //~ERROR `$s:stmt` is followed by `ident`
    ($s:stmt if) => {};       //~ERROR `$s:stmt` is followed by `if`
    ($s:stmt in) => {};       //~ERROR `$s:stmt` is followed by `in`
    ($s:stmt $p:pat) => {};   //~ERROR `$s:stmt` is followed by `$p:pat`
    ($s:stmt $e:expr) => {};  //~ERROR `$s:stmt` is followed by `$e:expr`
    ($s:stmt $t:ty) => {};    //~ERROR `$s:stmt` is followed by `$t:ty`
    ($s:stmt $t:stmt) => {};  //~ERROR `$s:stmt` is followed by `$t:stmt`
    ($s:stmt $p:path) => {};  //~ERROR `$s:stmt` is followed by `$p:path`
    ($s:stmt $b:block) => {}; //~ERROR `$s:stmt` is followed by `$b:block`
    ($s:stmt $i:ident) => {}; //~ERROR `$s:stmt` is followed by `$i:ident`
    ($s:stmt $t:tt) => {};    //~ERROR `$s:stmt` is followed by `$t:tt`
    ($s:stmt $i:item) => {};  //~ERROR `$s:stmt` is followed by `$i:item`
    ($s:stmt $m:meta) => {};  //~ERROR `$s:stmt` is followed by `$m:meta`
}
// FOLLOW(path) = FOLLOW(ty)
macro_rules! follow_path {
    ($p:path ()) => {};       //~ERROR  `$p:path` is followed by `(`
    ($p:path []) => {};       // ok (RFC 1462)
    ($p:path +) => {};        //~ERROR `$p:path` is followed by `+`
    ($p:path ident) => {};    //~ERROR `$p:path` is followed by `ident`
    ($p:path if) => {};       //~ERROR `$p:path` is followed by `if`
    ($p:path $q:pat) => {};   //~ERROR `$p:path` is followed by `$q:pat`
    ($p:path $e:expr) => {};  //~ERROR `$p:path` is followed by `$e:expr`
    ($p:path $t:ty) => {};    //~ERROR `$p:path` is followed by `$t:ty`
    ($p:path $s:stmt) => {};  //~ERROR `$p:path` is followed by `$s:stmt`
    ($p:path $q:path) => {};  //~ERROR `$p:path` is followed by `$q:path`
    ($p:path $b:block) => {}; // ok (RFC 1494)
    ($p:path $i:ident) => {}; //~ERROR `$p:path` is followed by `$i:ident`
    ($p:path $t:tt) => {};    //~ERROR `$p:path` is followed by `$t:tt`
    ($p:path $i:item) => {};  //~ERROR `$p:path` is followed by `$i:item`
    ($p:path $m:meta) => {};  //~ERROR `$p:path` is followed by `$m:meta`
}
// FOLLOW(block) = any token
// FOLLOW(ident) = any token

fn main() {}
