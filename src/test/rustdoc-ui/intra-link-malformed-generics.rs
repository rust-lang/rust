#![deny(broken_intra_doc_links)]

//! [Vec<] //~ ERROR
//! [Vec<Box<T] //~ ERROR
//! [Vec<Box<T>] //~ ERROR
//! [Vec<Box<T>>>] //~ ERROR
//! [Vec<T>>>] //~ ERROR
//! [<Vec] //~ ERROR
//! [Vec::<] //~ ERROR
//! [<T>] //~ ERROR
//! [<invalid syntax>] //~ ERROR
//! [Vec:<T>:new()] //~ ERROR
//! [Vec<<T>>] //~ ERROR
//! [Vec<>] //~ ERROR
//! [Vec<<>>] //~ ERROR

// FIXME(#74563) support UFCS
//! [<Vec as IntoIterator>::into_iter] //~ ERROR
//! [<Vec<T> as IntoIterator>::iter] //~ ERROR
