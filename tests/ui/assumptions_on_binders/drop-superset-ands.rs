//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]


// This test case not only tests superset filtering
// but also exercises the full `pull` flow
// (`pull_region_outlives_constraints_out_of_universe` method)
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
// Compared to the previous one, this test case is simpler and more direct
core::test_binder_constraints! {
    impl<'a: 'b, 'b, 'c: 'a> {
        forall<> {
            or {
                and { 'a: 'b, 'b: 'c },
                and { 'c: 'a },
                and { 'a: 'b },
            }
        } expect {
            or {
                and { 'c: 'a },
                and { 'a: 'b },
            }
        }
    }
}


fn main() {}
