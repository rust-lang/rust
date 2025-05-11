#![feature(box_patterns)]
#![warn(clippy::unnested_or_patterns)]
#![allow(
    clippy::cognitive_complexity,
    clippy::match_ref_pats,
    clippy::needless_if,
    clippy::manual_range_patterns
)]
#![allow(unreachable_patterns, irrefutable_let_patterns, unused_variables)]

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
    if let box (0 | 1) | (box 2 | box (3 | 4)) = Box::new(0) {}
    //~^ unnested_or_patterns
    if let box box 0 | box (box 2 | box 4) = Box::new(Box::new(0)) {}
    //~^ unnested_or_patterns
}
