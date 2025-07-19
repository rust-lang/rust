use std::rc::Rc;
use std::sync::Arc;
// Would work with perfect derives
#[derive(Clone, PartialEq)]
struct S<T, K, V>(Arc<T>, Rc<K>, Arc<V>, ());

// Wouldn't work with perfect derives, as T, K and V are used as fields directly
#[derive(Clone, PartialEq)]
struct N<T, K, V>(Arc<T>, T, Rc<K>, K, Arc<V>, V, ());

// Wouldn't work with perfect derives
#[derive(Clone, PartialEq)]
struct Z<T, K, V>(T, K, Option<V>);

struct X;
#[derive(Clone, PartialEq)]
struct Y;

fn foo() {
    let s = S(Arc::new(X), Rc::new(X), Arc::new(X), ());
    let s2 = S(Arc::new(X), Rc::new(X), Arc::new(X), ());
    // FIXME(estebank): the current diagnostics for `==` don't have the same amount of context.
    let _ = s == s2; //~ ERROR E0369
}
fn main() {
    let s = S(Arc::new(X), Rc::new(X), Arc::new(Y), ());
    let s2 = s.clone(); //~ ERROR E0599
    let s = S(Arc::new(X), Rc::new(X), Arc::new(X), ());
    let s2 = s.clone(); //~ ERROR E0599
    // FIXME(estebank): using `PartialEq::eq` instead of `==` because the later diagnostic doesn't
    // currently have the same amount of context.
    s.eq(s2); //~ ERROR E0599
    let n = N(Arc::new(X), X, Rc::new(Y), Y, Arc::new(X), X, ());
    let n2 = n.clone(); //~ ERROR E0599
    n.eq(n2); //~ ERROR E0599
    let z = Z(X, Y, Some(X));
    let z2 = z.clone(); //~ ERROR E0599
    n.eq(z2); //~ ERROR E0599
}
