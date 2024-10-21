//@ known-bug: #131295

#![feature(generic_const_exprs)]

async fn foo<'a>() -> [(); {
       let _y: &'a ();
       4
   }] {
}
