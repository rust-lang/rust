fn t7p<A, B, C>(f: impl Fn(B) -> C, g: impl Fn(A) -> B) -> impl Fn(A) -> C {
    move |a: A| -> C { f(g(a)) }
}

fn t8n<A, B, C>(f: impl Fn(A) -> B, g: impl Fn(A) -> C) -> impl Fn(A) -> (B, C)
where
    A: Copy,
{
    move |a: A| -> (B, C) {
        let b = a;
        let fa = f(a);
        let ga = g(b);
        (fa, ga)
    }
}

fn main() {
    let f = |(_, _)| {};
    let g = |(a, _)| a;
    let t7 = |env| |a| |b| t7p(f, g)(((env, a), b));
    let t8 = t8n(t7, t7p(f, g));
    //~^ ERROR: expected a `Fn<(_,)>` closure, found `impl Fn<(((_, _), _),)>
}
