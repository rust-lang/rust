// Unless we check for it, we can hide early-bound lifetime parameters on
// a function by using associated items. This is a bug.
//
// Regression test for <https://github.com/rust-lang/rust/issues/154490>

trait EvilTrait {
    type EvilAssoc<'a>;
    fn evil_assoc_1(_: Option<Self::EvilAssoc<'_>>) -> &i32 {
        todo!()
    }
}

// zero explicit generic lifetimes
fn evil_early_bound_1<T: EvilTrait>(_: Option<<T as EvilTrait>::EvilAssoc<'_>>) -> &i32 {
    todo!()
}

fn evil_early_bound_2<T: EvilTrait>(_: Option<<T as EvilTrait>::EvilAssoc<'_>>) -> &i32 {
    todo!()
}

fn evil_multi_bound_3<'b, T: EvilTrait>(
    _: Option<<T as EvilTrait>::EvilAssoc<'_>>
) -> (&i32, &'b i64) {
    todo!()
}

fn evil_early_bound_4<T: EvilTrait>(_: Option<<T as EvilTrait>::EvilAssoc<'_>>) -> i32 {
    todo!()
}

fn normal_early_bound<'eb: 'eb, T: EvilTrait>(_: &'eb i32 ) -> &'eb i32 {
    todo!()
}

fn normal_late_bound<'lb, T: EvilTrait>(_: &'lb i32 ) -> &'lb i32 {
    todo!()
}

struct LtWrapper<'a>(&'a i32);

fn elide_struct_1(_: &i32) -> LtWrapper {
    todo!()
}

fn elide_struct_2(_: &i32) -> LtWrapper {
    todo!()
}

fn foo<T: EvilTrait>() {
    static WHATEVER: i32 = 123;

    evil_early_bound_1::<'static, T>(None);
    //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    // ^ FIXME: should this have a better diagnostic?
    evil_early_bound_2::<T>(None);
    evil_multi_bound_3::<'static, 'static, T>(None);
    //~^ ERROR: function takes 1 lifetime argument but 2 lifetime arguments were supplied [E0107]
    evil_early_bound_4::<'static, T>(None);
    //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
    <T as EvilTrait>::evil_assoc_1::<'static>(None);
    //~^ ERROR: associated function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    elide_struct_1(&WHATEVER);
    elide_struct_2::<'static>(&WHATEVER);
    //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
    normal_early_bound::<'static, T>(&WHATEVER);
    // ^ this is fine
    normal_late_bound::<'static, T>(&WHATEVER);
    //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
}


fn main() {}
