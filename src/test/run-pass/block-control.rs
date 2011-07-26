// xfail-stage0

// I think that this is actually pretty workable as a scheme for using
// blocks to affect control flow. It is a little syntactically heavyweight
// right now, but a lot of that can be fixed. A more lightweight notation
// for blocks will go a long way. The other big problem is the excess of
// closing parens and braces, but that can mostly be fixed with some tweaks
// to the macros that are used to invoke iterators so that we could write
// something like:
// #iter (for_each(v), fn (&T y) -> flow[bool] { ... });

tag flow[T] {
    f_normal;
    f_break;
    f_continue;
    f_return(T);
}

// Right now we the parser doesn't allow macros to be defined at top level.
// However, they don't obey scope and so we can just stick them in a
// function and it will work out right...
fn bullshit_hack() {
#macro([#call_body(body),
        alt (body) {
            f_normal { }
            f_break { break; }
            f_continue { cont; }
            f_return(?x) { ret f_return(x); }
        }]);
#macro([#iter(e),
        alt (e) {
            f_normal { }
            f_break | f_continue { fail; }
            f_return(?x) { ret x; }
        }]);
#macro([#ret(e), {ret f_return(e);}]);
#macro([#cont(e), {ret f_continue;}]);
#macro([#break(e), {ret f_break;}]);
#macro([#body(body), {{body} ret f_normal;}]);
}

fn for_each[T,S](&T[] v, &block (&T) -> flow[S] f) -> flow[S] {
    for (T x in v) {
        #call_body (f(x));
    }
    ret f_normal;
}

fn contains[T](&T[] v, &T x) -> bool {
    #iter (for_each(v, block (&T y) -> flow[bool] { #body({
        if (x == y) { #ret (true); }
    })}));
    ret false;
}

fn map[T,S](&T[] v, &block (&T) -> S f) -> S[] {
    auto w = ~[];
    // We don't invoke the #iter macro because we can't force a return.
    for_each(v, block (&T x) -> flow[()] { #body({
        w += ~[f(x)];
    })});
    ret w;
}

fn log_int_vec(&int[] v) {
    for_each(v, block (&int i) -> flow[()] { #body({
        log_err i;
    })});
}

fn main() {
    auto v = ~[1,2,3,4,5,6,7];
    assert(contains(v, 7) == true);
    assert(contains(v, 2) == true);
    assert(contains(v, 0) == false);
    fn f(&int i) -> int { ret i*i; };
    auto w = map(v, f);
    assert(contains(w, 36) == true);
    assert(contains(w, 5) == false);
    log_int_vec(w);
}
