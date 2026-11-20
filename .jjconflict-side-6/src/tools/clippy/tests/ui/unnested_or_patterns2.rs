#![feature(deref_patterns)]
#![warn(clippy::unnested_or_patterns)]
#![allow(clippy::manual_range_patterns)]

fn main() {
    if let Some(Some(0)) | Some(Some(1)) = None {}
    //~^ unnested_or_patterns
    if let Some(Some(0)) | Some(Some(1) | Some(2)) = None {}
    //~^ unnested_or_patterns
    if let Some(Some(0 | 1) | Some(2)) | Some(Some(3) | Some(4)) = None {}
    //~^ unnested_or_patterns
    if let Some(Some(0) | Some(1 | 2)) = None {}
    //~^ unnested_or_patterns
    if let ((0,),) | ((1,) | (2,),) = ((0,),) {}
    //~^ unnested_or_patterns
    if let 0 | (1 | 2) = 0 {}
    //~^ unnested_or_patterns
    if let deref!(0 | 1) | (deref!(2) | deref!(3 | 4)) = Box::new(0) {}
    //~^ unnested_or_patterns
    if let deref!(deref!(0)) | deref!(deref!(2) | deref!(4)) = Box::new(Box::new(0)) {}
    //~^ unnested_or_patterns
}
