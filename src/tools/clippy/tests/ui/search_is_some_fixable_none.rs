//@run-rustfix
#![allow(dead_code, clippy::explicit_auto_deref, clippy::useless_vec)]
#![warn(clippy::search_is_some)]

fn main() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;

    // Check `find().is_none()`, single-line case.
    let _ = v.iter().find(|&x| *x < 0).is_none();
    let _ = (0..1).find(|x| **y == *x).is_none(); // one dereference less
    let _ = (0..1).find(|x| *x == 0).is_none();
    let _ = v.iter().find(|x| **x == 0).is_none();
    let _ = (4..5).find(|x| *x == 1 || *x == 3 || *x == 5).is_none();
    let _ = (1..3).find(|x| [1, 2, 3].contains(x)).is_none();
    let _ = (1..3).find(|x| *x == 0 || [1, 2, 3].contains(x)).is_none();
    let _ = (1..3).find(|x| [1, 2, 3].contains(x) || *x == 0).is_none();
    let _ = (1..3)
        .find(|x| [1, 2, 3].contains(x) || *x == 0 || [4, 5, 6].contains(x) || *x == -1)
        .is_none();

    // Check `position().is_none()`, single-line case.
    let _ = v.iter().position(|&x| x < 0).is_none();

    // Check `rposition().is_none()`, single-line case.
    let _ = v.iter().rposition(|&x| x < 0).is_none();

    let s1 = String::from("hello world");
    let s2 = String::from("world");

    // caller of `find()` is a `&`static str`
    let _ = "hello world".find("world").is_none();
    let _ = "hello world".find(&s2).is_none();
    let _ = "hello world".find(&s2[2..]).is_none();
    // caller of `find()` is a `String`
    let _ = s1.find("world").is_none();
    let _ = s1.find(&s2).is_none();
    let _ = s1.find(&s2[2..]).is_none();
    // caller of `find()` is slice of `String`
    let _ = s1[2..].find("world").is_none();
    let _ = s1[2..].find(&s2).is_none();
    let _ = s1[2..].find(&s2[2..]).is_none();
}

#[allow(clippy::clone_on_copy, clippy::map_clone)]
mod issue7392 {
    struct Player {
        hand: Vec<usize>,
    }
    fn filter() {
        let p = Player {
            hand: vec![1, 2, 3, 4, 5],
        };
        let filter_hand = vec![5];
        let _ = p
            .hand
            .iter()
            .filter(|c| filter_hand.iter().find(|cc| c == cc).is_none())
            .map(|c| c.clone())
            .collect::<Vec<_>>();
    }

    struct PlayerTuple {
        hand: Vec<(usize, char)>,
    }
    fn filter_tuple() {
        let p = PlayerTuple {
            hand: vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')],
        };
        let filter_hand = vec![5];
        let _ = p
            .hand
            .iter()
            .filter(|(c, _)| filter_hand.iter().find(|cc| c == *cc).is_none())
            .map(|c| c.clone())
            .collect::<Vec<_>>();
    }

    fn field_projection() {
        struct Foo {
            foo: i32,
            bar: u32,
        }
        let vfoo = vec![Foo { foo: 1, bar: 2 }];
        let _ = vfoo.iter().find(|v| v.foo == 1 && v.bar == 2).is_none();

        let vfoo = vec![(42, Foo { foo: 1, bar: 2 })];
        let _ = vfoo
            .iter()
            .find(|(i, v)| *i == 42 && v.foo == 1 && v.bar == 2)
            .is_none();
    }

    fn index_projection() {
        let vfoo = vec![[0, 1, 2, 3]];
        let _ = vfoo.iter().find(|a| a[0] == 42).is_none();
    }

    #[allow(clippy::match_like_matches_macro)]
    fn slice_projection() {
        let vfoo = vec![[0, 1, 2, 3, 0, 1, 2, 3]];
        let _ = vfoo.iter().find(|sub| sub[1..4].len() == 3).is_none();
    }

    fn please(x: &u32) -> bool {
        *x == 9
    }

    fn deref_enough(x: u32) -> bool {
        x == 78
    }

    fn arg_no_deref(x: &&u32) -> bool {
        **x == 78
    }

    fn more_projections() {
        let x = 19;
        let ppx: &u32 = &x;
        let _ = [ppx].iter().find(|ppp_x: &&&u32| please(**ppp_x)).is_none();
        let _ = [String::from("Hey hey")].iter().find(|s| s.len() == 2).is_none();

        let v = vec![3, 2, 1, 0];
        let _ = v.iter().find(|x| deref_enough(**x)).is_none();
        let _ = v.iter().find(|x: &&u32| deref_enough(**x)).is_none();

        #[allow(clippy::redundant_closure)]
        let _ = v.iter().find(|x| arg_no_deref(x)).is_none();
        #[allow(clippy::redundant_closure)]
        let _ = v.iter().find(|x: &&u32| arg_no_deref(x)).is_none();
    }

    fn field_index_projection() {
        struct FooDouble {
            bar: Vec<Vec<i32>>,
        }
        struct Foo {
            bar: Vec<i32>,
        }
        struct FooOuter {
            inner: Foo,
            inner_double: FooDouble,
        }
        let vfoo = vec![FooOuter {
            inner: Foo { bar: vec![0, 1, 2, 3] },
            inner_double: FooDouble {
                bar: vec![vec![0, 1, 2, 3]],
            },
        }];
        let _ = vfoo
            .iter()
            .find(|v| v.inner_double.bar[0][0] == 2 && v.inner.bar[0] == 2)
            .is_none();
    }

    fn index_field_projection() {
        struct Foo {
            bar: i32,
        }
        struct FooOuter {
            inner: Vec<Foo>,
        }
        let vfoo = vec![FooOuter {
            inner: vec![Foo { bar: 0 }],
        }];
        let _ = vfoo.iter().find(|v| v.inner[0].bar == 2).is_none();
    }

    fn double_deref_index_projection() {
        let vfoo = vec![&&[0, 1, 2, 3]];
        let _ = vfoo.iter().find(|x| (**x)[0] == 9).is_none();
    }

    fn method_call_by_ref() {
        struct Foo {
            bar: u32,
        }
        impl Foo {
            pub fn by_ref(&self, x: &u32) -> bool {
                *x == self.bar
            }
        }
        let vfoo = vec![Foo { bar: 1 }];
        let _ = vfoo.iter().find(|v| v.by_ref(&v.bar)).is_none();
    }

    fn ref_bindings() {
        let _ = [&(&1, 2), &(&3, 4), &(&5, 4)].iter().find(|(&x, y)| x == *y).is_none();
        let _ = [&(&1, 2), &(&3, 4), &(&5, 4)].iter().find(|&(&x, y)| x == *y).is_none();
    }

    fn test_string_1(s: &str) -> bool {
        s.is_empty()
    }

    fn test_u32_1(s: &u32) -> bool {
        s.is_power_of_two()
    }

    fn test_u32_2(s: u32) -> bool {
        s.is_power_of_two()
    }

    fn projection_in_args_test() {
        // Index projections
        let lst = &[String::from("Hello"), String::from("world")];
        let v: Vec<&[String]> = vec![lst];
        let _ = v.iter().find(|s| s[0].is_empty()).is_none();
        let _ = v.iter().find(|s| test_string_1(&s[0])).is_none();

        // Field projections
        struct FieldProjection<'a> {
            field: &'a u32,
        }
        let field = 123456789;
        let instance = FieldProjection { field: &field };
        let v = vec![instance];
        let _ = v.iter().find(|fp| fp.field.is_power_of_two()).is_none();
        let _ = v.iter().find(|fp| test_u32_1(fp.field)).is_none();
        let _ = v.iter().find(|fp| test_u32_2(*fp.field)).is_none();
    }
}
