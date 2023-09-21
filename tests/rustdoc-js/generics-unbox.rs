pub struct Out<A, B = ()> {
    a: A,
    b: B,
}

pub struct Out1<A, const N: usize> {
    a: [A; N],
}

pub struct Out2<A, const N: usize> {
    a: [A; N],
}

pub struct Out3<A, B> {
    a: A,
    b: B,
}

pub struct Out4<A, B> {
    a: A,
    b: B,
}

pub struct Inside<T>(T);

pub fn alpha<const N: usize, T>(_: Inside<T>) -> Out<Out1<T, N>, Out2<T, N>> {
    loop {}
}

pub fn beta<T, U>(_: Inside<T>) -> Out<Out3<T, U>, Out4<U, T>> {
    loop {}
}

pub fn gamma<T, U>(_: Inside<T>) -> Out<Out3<U, T>, Out4<T, U>> {
    loop {}
}
