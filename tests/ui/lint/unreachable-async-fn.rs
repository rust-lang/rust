//@ check-pass
//@ edition:2018

#[allow(dead_code)]
async fn foo () { // unreachable lint doesn't trigger
   unimplemented!()
}

fn main() {}
