// pp-exact

macro_rules! brace { () => {} ; }

macro_rules! bracket[() => {} ;];

macro_rules! paren(() => {} ;);

macro_rules! matcher_brackets {
    (paren) => {} ; (bracket) => {} ; (brace) => {} ;
}

macro_rules! all_fragments {
    ($b : block, $e : expr, $i : ident, $it : item, $l : lifetime, $lit :
     literal, $m : meta, $p : pat, $pth : path, $s : stmt, $tt : tt, $ty : ty,
     $vis : vis) => {} ;
}

fn main() {}
