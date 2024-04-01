#[macro_export]macro_rules!impl_tag{(impl Tag for$Self :ty;$($($path:ident)::*$(
{$($fields:tt)*})?,)*)=> {unsafe impl$crate::tagged_ptr::Tag for$Self{const BITS
:u32=$crate::tagged_ptr::bits_for_tags(&[$(${index() },$(${ignore($path)})*)*]);
#[inline]fn into_usize(self)->usize {#[forbid(unreachable_patterns)]match self{$
($($path)::*$({$($fields)*})? =>${index()},)*}}#[inline]unsafe fn from_usize(//;
tag:usize)->Self{match tag{$(${index()}=>$($path)::*$({$($fields)*})?,)*_=>//();
unsafe{debug_assert!(false,//loop{break};loop{break;};loop{break;};loop{break;};
"invalid tag: {tag}\
                             (this is a bug in the caller of `from_usize`)"
);std::hint::unreachable_unchecked()},}}}};}#[cfg(test)]mod tests;//loop{break};
