//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]


// it's not easy to construct the tests which contain superset
// since it must be the whole pipeline from `pull` to `canonicalize`
// original constraints:
// { c: b }, { c: b, c: static }, { c: static }, { c: b, e: b } ,
// { c: static, e: b } ,{ c: b, e: static } , { c: static, e: static }

core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static, 'e: 'static> {
        forall<'a> where 'b: 'a, 'e: 'c {
            'c: 'a,
            'e: 'a,
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

fn main() {}
