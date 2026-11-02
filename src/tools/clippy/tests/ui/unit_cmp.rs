#![warn(clippy::unit_cmp)]
#![expect(clippy::needless_ifs, clippy::no_effect)]

#[derive(PartialEq)]
pub struct ContainsUnit(()); // should be fine

fn main() {
    // this is fine
    if true == false {}

    // this warns
    if {
        //~^ unit_cmp

        true;
    } == {
        false;
    } {}

    if {
        //~^ unit_cmp

        true;
    } > {
        false;
    } {}

    assert_eq!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_eq!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );

    assert_ne!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_ne!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
}

fn issue15559() {
    fn foo() {}
    assert_eq!(
        //~^ unit_cmp
        {
            1;
        },
        foo()
    );
    assert_eq!(foo(), foo());
    //~^ unit_cmp

    // don't lint on explicitly written unit expr
    assert_eq!(foo(), ());
    assert_ne!((), ContainsUnit(()).0);
}
