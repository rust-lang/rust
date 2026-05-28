// Regression test for #72051, hang when resolving regions.

//@ check-pass
//@ edition:2018

pub async fn query<'a>(_: &(), _: &(), _: (&(dyn std::any::Any + 'a),) ) {}
fn main() {}
