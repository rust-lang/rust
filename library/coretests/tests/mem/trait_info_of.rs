use std::any::TypeId;
use std::ptr::DynMetadata;

struct Garlic(i32);
trait Blah {
    fn get_truth(&self) -> i32;
}
impl Blah for Garlic {
    fn get_truth(&self) -> i32 {
        self.0 * 21
    }
}

#[test]
fn test_implements_trait() {
    const {
        assert!(TypeId::of::<Garlic>().trait_info_of::<dyn Blah>().is_some());
        assert!(TypeId::of::<Garlic>().trait_info_of::<dyn Blah + Send>().is_some());
        assert!(TypeId::of::<*const Box<Garlic>>().trait_info_of::<dyn Sync>().is_none());
        assert!(TypeId::of::<u8>().trait_info_of_trait_type_id(TypeId::of::<dyn Blah>()).is_none());
    }
}

#[test]
fn test_dyn_creation() {
    let garlic = Garlic(2);
    unsafe {
        assert_eq!(
            std::ptr::from_raw_parts::<dyn Blah>(
                &raw const garlic,
                const { TypeId::of::<Garlic>().trait_info_of::<dyn Blah>() }.unwrap().get_vtable()
            )
            .as_ref()
            .unwrap()
            .get_truth(),
            42
        );
    }

    assert_eq!(
        const {
            TypeId::of::<Garlic>()
            .trait_info_of_trait_type_id(TypeId::of::<dyn Blah>())
            .unwrap()
        }.get_vtable(),
        unsafe {
            crate::mem::transmute::<_, DynMetadata<*const ()>>(
                const {
                    TypeId::of::<Garlic>().trait_info_of::<dyn Blah>()
                }.unwrap().get_vtable(),
            )
        }
    );
}

#[test]
fn test_incorrect_use() {
    assert_eq!(
        const { TypeId::of::<i32>().trait_info_of_trait_type_id(TypeId::of::<u32>()) },
        None
    );
}

trait DstTrait {}
impl DstTrait for [i32] {}

#[test]
fn dst_ice() {
    assert!(const { TypeId::of::<[i32]>().trait_info_of::<dyn DstTrait>() }.is_none());
}
