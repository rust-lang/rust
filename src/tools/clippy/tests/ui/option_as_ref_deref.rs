#![allow(unused, clippy::redundant_clone, clippy::useless_vec)]
#![warn(clippy::option_as_ref_deref)]

use std::ffi::{CString, OsString};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

fn main() {
    let mut opt = Some(String::from("123"));

    let _ = opt.clone().as_ref().map(Deref::deref).map(str::len);
    //~^ option_as_ref_deref

    #[rustfmt::skip]
    let _ = opt.clone()
    //~^ option_as_ref_deref
        .as_ref().map(
            Deref::deref
        )
        .map(str::len);

    let _ = opt.as_mut().map(DerefMut::deref_mut);
    //~^ option_as_ref_deref

    let _ = opt.as_ref().map(String::as_str);
    //~^ option_as_ref_deref
    let _ = opt.as_ref().map(|x| x.as_str());
    //~^ option_as_ref_deref
    let _ = opt.as_mut().map(String::as_mut_str);
    //~^ option_as_ref_deref
    let _ = opt.as_mut().map(|x| x.as_mut_str());
    //~^ option_as_ref_deref
    let _ = Some(CString::new(vec![]).unwrap()).as_ref().map(CString::as_c_str);
    //~^ option_as_ref_deref
    let _ = Some(OsString::new()).as_ref().map(OsString::as_os_str);
    //~^ option_as_ref_deref
    let _ = Some(PathBuf::new()).as_ref().map(PathBuf::as_path);
    //~^ option_as_ref_deref
    let _ = Some(Vec::<()>::new()).as_ref().map(Vec::as_slice);
    //~^ option_as_ref_deref
    let _ = Some(Vec::<()>::new()).as_mut().map(Vec::as_mut_slice);
    //~^ option_as_ref_deref

    let _ = opt.as_ref().map(|x| x.deref());
    //~^ option_as_ref_deref
    let _ = opt.clone().as_mut().map(|x| x.deref_mut()).map(|x| x.len());
    //~^ option_as_ref_deref

    let vc = vec![String::new()];
    let _ = Some(1_usize).as_ref().map(|x| vc[*x].as_str()); // should not be linted

    let _: Option<&str> = Some(&String::new()).as_ref().map(|x| x.as_str()); // should not be linted

    let _ = opt.as_ref().map(|x| &**x);
    //~^ option_as_ref_deref
    let _ = opt.as_mut().map(|x| &mut **x);
    //~^ option_as_ref_deref

    // Issue #5927
    let _ = opt.as_ref().map(std::ops::Deref::deref);
    //~^ option_as_ref_deref
}

#[clippy::msrv = "1.39"]
fn msrv_1_39() {
    let opt = Some(String::from("123"));
    let _ = opt.as_ref().map(String::as_str);
}

#[clippy::msrv = "1.40"]
fn msrv_1_40() {
    let opt = Some(String::from("123"));
    let _ = opt.as_ref().map(String::as_str);
    //~^ option_as_ref_deref
}
