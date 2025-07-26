#![feature(sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

pub trait SizedTr {}

impl<T: Sized> SizedTr for T {}

pub trait NegSizedTr {}

impl<T: ?Sized> NegSizedTr for T {}

pub trait MetaSizedTr {}

impl<T: MetaSized> MetaSizedTr for T {}

pub trait PointeeSizedTr: PointeeSized {}

impl<T: PointeeSized> PointeeSizedTr for T {}
