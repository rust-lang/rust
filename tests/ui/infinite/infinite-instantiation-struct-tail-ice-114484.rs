//~ ERROR reached the recursion limit finding the struct tail for `[u8; 256]`
//~| ERROR reached the recursion limit finding the struct tail for `[u8; 256]`
//~| ERROR reached the recursion limit finding the struct tail for `[u8; 256]`
//~| ERROR reached the recursion limit finding the struct tail for `[u8; 256]`
//~| ERROR reached the recursion limit finding the struct tail for `SomeData<256>`
//~| ERROR reached the recursion limit finding the struct tail for `SomeData<256>`
//~| ERROR reached the recursion limit finding the struct tail for `SomeData<256>`
//~| ERROR reached the recursion limit finding the struct tail for `SomeData<256>`
//~| ERROR reached the recursion limit finding the struct tail for `VirtualWrapper<SomeData<256>, 0>`
//~| ERROR reached the recursion limit finding the struct tail for `VirtualWrapper<SomeData<256>, 0>`
//~| ERROR reached the recursion limit finding the struct tail for `VirtualWrapper<SomeData<256>, 0>`
//~| ERROR reached the recursion limit finding the struct tail for `VirtualWrapper<SomeData<256>, 0>`
//~| ERROR reached the recursion limit while instantiating `<VirtualWrapper<..., 1> as MyTrait>::virtualize`

//@ build-fail
//@ compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

// Regression test for #114484: This used to ICE during monomorphization, because we treated
// `<VirtualWrapper<...> as Pointee>::Metadata` as a rigid projection after reaching the recursion
// limit when finding the struct tail.

use std::marker::PhantomData;

trait MyTrait {
    fn virtualize(&self) -> &dyn MyTrait;
}

struct VirtualWrapper<T, const L: u8>(T);

impl<T, const L: u8> VirtualWrapper<T, L> {
    pub fn wrap(value: &T) -> &Self {
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<T: MyTrait + 'static, const L: u8> MyTrait for VirtualWrapper<T, L> {
    fn virtualize(&self) -> &dyn MyTrait {
        unsafe { virtualize_my_trait(L, self) }
        // unsafe { virtualize_my_trait(L, &self.0) } // <-- this code fixes the problem
    }
}

const unsafe fn virtualize_my_trait<T>(level: u8, obj: &T) -> &dyn MyTrait
where
    T: MyTrait + 'static,
{
    const fn gen_vtable_ptr<T, const L: u8>() -> *const ()
    where
        T: MyTrait + 'static,
    {
        let [_, vtable] = unsafe {
            std::mem::transmute::<*const dyn MyTrait, [*const (); 2]>(std::ptr::null::<
                VirtualWrapper<T, L>,
            >())
        };
        vtable
    }

    struct Vtable<T>(PhantomData<T>);

    impl<T> Vtable<T>
    where
        T: MyTrait + 'static,
    {
        const LEVELS: [*const (); 2] = [gen_vtable_ptr::<T, 1>(), gen_vtable_ptr::<T, 2>()];
    }

    let vtable = Vtable::<T>::LEVELS[(level != 0) as usize];

    let data = obj as *const T as *const ();
    let ptr: *const dyn MyTrait = std::mem::transmute([data, vtable]);

    &*ptr
}

struct SomeData<const N: usize>([u8; N]);

impl<const N: usize> MyTrait for SomeData<N> {
    fn virtualize(&self) -> &dyn MyTrait {
        VirtualWrapper::<Self, 0>::wrap(self)
    }
}

fn main() {
    let test = SomeData([0; 256]);
    test.virtualize();
}
