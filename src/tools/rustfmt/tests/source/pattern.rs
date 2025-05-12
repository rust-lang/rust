// rustfmt-normalize_comments: true
#![feature(exclusive_range_pattern)]
use core::u8::MAX;

fn main() {
    let z = match x {
        "pat1" => 1,
        ( ref  x, ref  mut  y /*comment*/) => 2,
    };

    if let <  T as  Trait   > :: CONST = ident {
        do_smth();
    }

    let Some ( ref   xyz  /*   comment!   */) = opt;

    if let  None  =   opt2 { panic!("oh noes"); }

    let foo@bar (f) = 42;
    let a::foo ( ..) = 42;
    let [ ] = 42;
    let [a,     b,c ] = 42;
    let [ a,b,c ] = 42;
    let [a,    b, c, d,e,f,     g] = 42;
    let foo {   } = 42;
    let foo {..} = 42;
    let foo { x, y: ref foo,     .. } = 42;
    let foo { x, yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: ref foo,     .. } = 42;
    let foo { x,       yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: ref foo,      } = 42;
    let foo { x, yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: ref foo,     .. };
    let foo { x,       yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: ref foo,      };

    match b"12" {
        [0,
        1..MAX
        ] => {}
        _ => {}
    }
}

impl<'a,'b> ResolveGeneratedContentFragmentMutator<'a,'b> {
    fn mutate_fragment(&mut self, fragment: &mut Fragment) {
        match **info {
            GeneratedContentInfo::ContentItem(
                ContentItem::Counter(
                    ref counter_name,
                    counter_style
                )
            ) => {}}}
}

fn issue_1319() {
    if let (Event { .. }, .. ) = ev_state {}
}

fn issue_1874() {
    if let Some(()) = x {
y
    }
}

fn combine_patterns() {
    let x = match y {
        Some(
            Some(
                Foo {
                    z: Bar(..),
                    a: Bar(..),
                    b: Bar(..),
                },
            ),
        ) => z,
        _ => return,
    };
}

fn slice_patterns() {
    match b"123" {
        [0, ..] => {}
        [0, foo] => {}
        _ => {}
    }
}

fn issue3728() {
    let foo = |
    (c,)
        | c;
    foo((1,));
}

fn literals() {
    match 42 {
        1 | 2 | 4
        | 6 => {}
        10 | 11 | 12
        | 13 | 14 => {}
        _ => {}
    }
}