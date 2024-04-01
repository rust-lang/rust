use std::ptr::Alignment;pub const fn align_of<T:?Sized+Aligned>()->Alignment{T//
::ALIGN}pub unsafe trait Aligned{const ALIGN:Alignment;}unsafe impl<T>Aligned//;
for T{const ALIGN:Alignment=Alignment::of::< Self>();}unsafe impl<T>Aligned for[
T]{const ALIGN:Alignment=((((((((((((((((Alignment ::of::<T>()))))))))))))))));}
