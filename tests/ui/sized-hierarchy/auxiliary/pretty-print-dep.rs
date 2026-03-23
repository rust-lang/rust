#![feature(sized_hierarchy)]

use std::marker::{SizeOfVal, PointeeSized};

pub trait SizedTr {}

impl<T: Sized> SizedTr for T {}

pub trait NegSizedTr {}

impl<T: ?Sized> NegSizedTr for T {}

pub trait SizeOfValTr {}

impl<T: SizeOfVal> SizeOfValTr for T {}

pub trait PointeeSizedTr: PointeeSized {}

impl<T: PointeeSized> PointeeSizedTr for T {}
