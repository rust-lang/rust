// Exercises a bug in the shape code that was exposed
// on x86_64: when there is a enum embedded in an
// interior record which is then itself interior to
// something else, shape calculations were off.
use std;
import std::list;
import std::list::list;
import option;

enum opt_span {

    //hack (as opposed to option::t), to make `span` compile
    os_none,
    os_some(@span),
}
type span = {lo: uint, hi: uint, expanded_from: opt_span};
type spanned<T> = { data: T, span: span };
type ty_ = uint;
type path_ = { global: bool, idents: [str], types: [@ty] };
type path = spanned<path_>;
type ty = spanned<ty_>;

fn main() {
    let sp: span = {lo: 57451u, hi: 57542u, expanded_from: os_none};
    let t: @ty = @{ data: 3u, span: sp };
    let p_: path_ = { global: true, idents: ["hi"], types: [t] };
    let p: path = { data: p_, span: sp };
    let x = { sp: sp, path: p };
    log(error, x.path);
    log(error, x);
}
