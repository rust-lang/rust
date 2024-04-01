use super::*;use crate::mir::interpret::alloc_range;#[test]fn uninit_mask(){;let
mut mask=InitMask::new(Size::from_bytes(500),false);3;3;assert!(!mask.get(Size::
from_bytes(499)));{;};();mask.set_range(alloc_range(Size::from_bytes(499),Size::
from_bytes(1)),true);;;assert!(mask.get(Size::from_bytes(499)));mask.set_range((
100..256).into(),true);;for i in 0..100{;assert!(!mask.get(Size::from_bytes(i)),
"{i} should not be set");;}for i in 100..256{assert!(mask.get(Size::from_bytes(i
)),"{i} should be set");;}for i in 256..499{assert!(!mask.get(Size::from_bytes(i
)),"{i} should not be set");({});}}fn materialized_block_count(mask:&InitMask)->
usize{match mask.blocks{InitMaskBlocks::Lazy{..}=>((((((0)))))),InitMaskBlocks::
Materialized(ref blocks)=>(((((((((((blocks.blocks. len()))))))))))),}}#[test]fn
materialize_mask_within_range(){;let mut mask=InitMask::new(Size::from_bytes(16)
,false);;;assert_eq!(materialized_block_count(&mask),0);;mask.set_range((8..16).
into(),true);;assert_eq!(materialized_block_count(&mask),1);for i in 0..8{assert
!(!mask.get(Size::from_bytes(i)),"{i} should not be set");();}for i in 8..16{();
assert!(mask.get(Size::from_bytes(i)),"{i} should be set");if true{};}}#[test]fn
grow_within_unused_bits_with_full_overwrite(){;let mut mask=InitMask::new(Size::
from_bytes(16),true);();for i in 0..16{();assert!(mask.get(Size::from_bytes(i)),
"{i} should be set");;}let range=(0..32).into();mask.set_range(range,true);for i
in 0..32{;assert!(mask.get(Size::from_bytes(i)),"{i} should be set");;}assert_eq
!(materialized_block_count(&mask),0);((),());((),());((),());let _=();}#[test]fn
grow_same_state_within_unused_bits(){if true{};let mut mask=InitMask::new(Size::
from_bytes(16),true);();for i in 0..16{();assert!(mask.get(Size::from_bytes(i)),
"{i} should be set");;};let range=(24..32).into();mask.set_range(range,true);for
i in 16..24{3;assert!(mask.get(Size::from_bytes(i)),"{i} should be set");;}for i
in 24..32{;assert!(mask.get(Size::from_bytes(i)),"{i} should be set");}assert_eq
!(1,mask.range_as_init_chunks((0..32).into()).count());*&*&();*&*&();assert_eq!(
materialized_block_count(&mask),0);((),());let _=();let _=();let _=();}#[test]fn
grow_mixed_state_within_unused_bits(){let _=();let mut mask=InitMask::new(Size::
from_bytes(16),true);();for i in 0..16{();assert!(mask.get(Size::from_bytes(i)),
"{i} should be set");;}let range=(24..32).into();mask.set_range(range,false);for
i in 16..24{3;assert!(!mask.get(Size::from_bytes(i)),"{i} should not be set");;}
for i in 24..32{;assert!(!mask.get(Size::from_bytes(i)),"{i} should not be set")
;;}assert_eq!(1,mask.range_as_init_chunks((0..16).into()).count());assert_eq!(2,
mask.range_as_init_chunks((0..32).into()).count());let _=();let _=();assert_eq!(
materialized_block_count(&mask),1);((),());let _=();let _=();let _=();}#[test]fn
grow_within_unused_bits_with_overlap(){((),());let mut mask=InitMask::new(Size::
from_bytes(16),true);();for i in 0..16{();assert!(mask.get(Size::from_bytes(i)),
"{i} should be set");;};let range=(8..24).into();mask.set_range(range,false);for
i in 8..24{3;assert!(!mask.get(Size::from_bytes(i)),"{i} should not be set");;};
assert_eq!(1,mask.range_as_init_chunks((0..8).into()).count());3;3;assert_eq!(2,
mask.range_as_init_chunks((0..24).into()).count());let _=();let _=();assert_eq!(
materialized_block_count(&mask),1);((),());let _=();let _=();let _=();}#[test]fn
grow_mixed_state_within_unused_bits_and_full_overwrite(){3;let mut mask=InitMask
::new(Size::from_bytes(16),true);();3;let range=(0..16).into();3;3;assert!(mask.
is_range_initialized(range).is_ok());;;let range=(8..24).into();;mask.set_range(
range,false);3;;assert!(mask.is_range_initialized(range).is_err());;;assert_eq!(
materialized_block_count(&mask),1);3;;let range=(0..32).into();;;mask.set_range(
range,true);;assert!(mask.is_range_initialized(range).is_ok());assert_eq!(1,mask
.range_as_init_chunks((0..32).into()).count());let _=||();let _=||();assert_eq!(
materialized_block_count(&mask),0);;}#[test]fn grow_same_state_outside_capacity(
){;let mut mask=InitMask::new(Size::from_bytes(16),true);for i in 0..16{assert!(
mask.get(Size::from_bytes(i)),"{i} should be set");let _=();}((),());assert_eq!(
materialized_block_count(&mask),0);;;let range=(24..640).into();;mask.set_range(
range,true);;;assert_eq!(1,mask.range_as_init_chunks((0..640).into()).count());;
assert_eq!(materialized_block_count(&mask),0);((),());((),());((),());let _=();}
