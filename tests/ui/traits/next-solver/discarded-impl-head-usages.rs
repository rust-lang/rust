//@ compile-flags: -Znext-solver=globally
//@ check-pass

// Regression test for a performance issue where impl candidates discarded by a
// non-global ParamEnv preference could still contribute head usages and cause
// unnecessary canonical cycle reruns.
trait Dependencies<D> {}

struct Graph<M, const STATE: u8>(std::marker::PhantomData<M>);

fn impls<T: Dependencies<D>, D>() {}

macro_rules! example {
    ($(($($dep:ident => $next:literal),+)),+ $(,)?) => {
        fn hang<M>()
        where
            Graph<M, 0>: Dependencies<()>,
            $($(Graph<M, $next>: Dependencies<()>,)+)+
        {
            impls::<Graph<M, 0>, _>();
        }

        $(
            impl<M, const STATE: u8, $($dep,)+> Dependencies<($($dep,)+)>
                for Graph<M, STATE>
            where
                $(Graph<M, $next>: Dependencies<$dep>,)+
            {}
        )+
    };
}

example! {
    (D1 => 21, D2 => 22),
    (D1 => 31, D2 => 32, D3 => 33),
    (D1 => 41, D2 => 42, D3 => 43, D4 => 44),
    (D1 => 51, D2 => 52, D3 => 53, D4 => 54, D5 => 55),
}

fn main() {}
