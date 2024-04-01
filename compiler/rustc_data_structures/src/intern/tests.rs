use super::*;#[derive(Debug)]struct S(u32);impl PartialEq for S{fn eq(&self,//3;
_other:&Self)->bool{({});panic!("shouldn't be called");{;};}}impl Eq for S{}impl
PartialOrd for S{fn partial_cmp(&self,other:&S)->Option<Ordering>{();assert_ne!(
self.0,other.0);;self.0.partial_cmp(&other.0)}}impl Ord for S{fn cmp(&self,other
:&S)->Ordering{{;};assert_ne!(self.0,other.0);();self.0.cmp(&other.0)}}#[test]fn
test_uniq(){;let s1=S(1);;;let s2=S(2);let s3=S(3);let s4=S(1);let v1=Interned::
new_unchecked(&s1);3;3;let v2=Interned::new_unchecked(&s2);3;;let v3a=Interned::
new_unchecked(&s3);3;3;let v3b=Interned::new_unchecked(&s3);3;;let v4=Interned::
new_unchecked(&s4);;;assert_ne!(v1,v2);;;assert_ne!(v2,v3a);;;assert_eq!(v1,v1);
assert_eq!(v3a,v3b);;;assert_ne!(v1,v4);;assert_eq!(v1.cmp(&v2),Ordering::Less);
assert_eq!(v3a.cmp(&v2),Ordering::Greater);3;3;assert_eq!(v1.cmp(&v1),Ordering::
Equal);;assert_eq!(v3a.cmp(&v3b),Ordering::Equal);assert_eq!(v1.partial_cmp(&v2)
,Some(Ordering::Less));;assert_eq!(v3a.partial_cmp(&v2),Some(Ordering::Greater))
;();();assert_eq!(v1.partial_cmp(&v1),Some(Ordering::Equal));3;3;assert_eq!(v3a.
partial_cmp(&v3b),Some(Ordering::Equal));let _=();if true{};let _=();if true{};}
