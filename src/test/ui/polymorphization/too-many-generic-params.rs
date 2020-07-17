// build-pass
#![feature(rustc_attrs)]

// This test checks that the analysis doesn't panic when there are >64 generic parameters, but
// instead considers those parameters used.

#[rustc_polymorphize_error]
fn bar<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, AA,
       AB, AC, AD, AE, AF, AG, AH, AI, AJ, AK, AL, AM, AN, AO, AP, AQ, AR, AS, AT, AU, AV, AW,
       AX, AY, AZ, BA, BB, BC, BD, BE, BF, BG, BH, BI, BJ, BK, BL, BM>()
{
    let _: Option<A> = None;
    let _: Option<B> = None;
    let _: Option<C> = None;
    let _: Option<D> = None;
    let _: Option<E> = None;
    let _: Option<F> = None;
    let _: Option<G> = None;
    let _: Option<H> = None;
    let _: Option<I> = None;
    let _: Option<J> = None;
    let _: Option<K> = None;
    let _: Option<L> = None;
    let _: Option<M> = None;
    let _: Option<N> = None;
    let _: Option<O> = None;
    let _: Option<P> = None;
    let _: Option<Q> = None;
    let _: Option<R> = None;
    let _: Option<S> = None;
    let _: Option<T> = None;
    let _: Option<U> = None;
    let _: Option<V> = None;
    let _: Option<W> = None;
    let _: Option<X> = None;
    let _: Option<Y> = None;
    let _: Option<Z> = None;
    let _: Option<AA> = None;
    let _: Option<AB> = None;
    let _: Option<AC> = None;
    let _: Option<AD> = None;
    let _: Option<AE> = None;
    let _: Option<AF> = None;
    let _: Option<AG> = None;
    let _: Option<AH> = None;
    let _: Option<AI> = None;
    let _: Option<AJ> = None;
    let _: Option<AK> = None;
    let _: Option<AL> = None;
    let _: Option<AM> = None;
    let _: Option<AN> = None;
    let _: Option<AO> = None;
    let _: Option<AP> = None;
    let _: Option<AQ> = None;
    let _: Option<AR> = None;
    let _: Option<AS> = None;
    let _: Option<AT> = None;
    let _: Option<AU> = None;
    let _: Option<AV> = None;
    let _: Option<AW> = None;
    let _: Option<AX> = None;
    let _: Option<AY> = None;
    let _: Option<AZ> = None;
    let _: Option<BA> = None;
    let _: Option<BB> = None;
    let _: Option<BC> = None;
    let _: Option<BD> = None;
    let _: Option<BE> = None;
    let _: Option<BF> = None;
    let _: Option<BG> = None;
    let _: Option<BH> = None;
    let _: Option<BI> = None;
    let _: Option<BJ> = None;
    let _: Option<BK> = None;
    let _: Option<BL> = None;
    let _: Option<BM> = None;
}

fn main() {
    bar::<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
          u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
          u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
          u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
          u32>();
}
