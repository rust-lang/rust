//@ known-bug: #23707
//@ compile-flags: -Copt-level=0 --edition=2021
//@ only-x86_64
#![recursion_limit="2048"]

use std::marker::PhantomData;
use std::fmt;
use std::fmt::Debug;

pub struct Z( () );
pub struct S<T> (PhantomData<T>);


pub trait Nat {
    fn sing() -> Self;
    fn get(&self) -> usize;
}

impl Nat for Z {
    fn sing() -> Z { Z( () ) }
    #[inline(always)]
    fn get(&self) -> usize {
        0
    }
}

impl<T : Nat> Nat for S<T> {
    fn sing() -> S<T> { S::<T>( PhantomData::<T> ) }
    #[inline(always)]
    fn get(&self) -> usize {
        let prd : T = Nat::sing();
        1 + prd.get()
    }
}

pub type N0 = Z;
pub type N1 = S<N0>;
pub type N2 = S<N1>;
pub type N3 = S<N2>;
pub type N4 = S<N3>;
pub type N5 = S<N4>;


pub struct Node<D : Nat>(usize,PhantomData<D>);

impl<D:Nat> Node<D> {
    pub fn push(&self, c : usize) -> Node<S<D>> {
        let Node(i,_) = *self;
        Node(10*i+c, PhantomData::<S<D>>)
    }
}

impl<D:Nat> Node<S<D>> {
    pub fn pop(&self) -> (Node<D>,usize) {
        let Node(i,_) = *self;
        (Node(i/10, PhantomData::<D>), i-10*(i/10))
    }
}

impl<D:Nat> Debug for Node<D> {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        let s : D = Nat::sing();
        write!(f, "Node<{}>: i= {}",
               s.get(), self.0)
    }
}
pub trait Step {
    fn step(&self, usize) -> Self;
}

impl Step for Node<N0> {
    #[inline(always)]
    fn step(&self, n : usize) -> Node<N0> {
        println!("base case");
        Node(n,PhantomData::<N0>)
    }
}

impl<D:Nat> Step for Node<S<D>>
    where Node<D> : Step {
        #[inline(always)]
        fn step(&self, n : usize) -> Node<S<D>> {
            println!("rec");
            let (par,c) = self.pop();
            let cnew = c+n;
            par.step(c).push(cnew)
        }

}

fn tst<D:Nat>(ref p : &Node<D>, c : usize) -> usize
    where Node<D> : Step {
        let Node(i,_) = p.step(c);
        i
}



fn main() {
    let nd : Node<N3> = Node(555,PhantomData::<N3>);

    // overflow...core::marker::Size
    let Node(g,_) = tst(nd,1);

    // ok
    //let Node(g,_) = nd.step(1);

    println!("{:?}", g);
}
