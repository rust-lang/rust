use super::{TocBuilder, Toc, TocEntry};

#[test]
fn builder_smoke() {
    let mut builder = TocBuilder::new();

    // this is purposely not using a fancy macro like below so
    // that we're sure that this is doing the correct thing, and
    // there's been no macro mistake.
    macro_rules! push {
        ($level: expr, $name: expr) => {
            assert_eq!(builder.push($level,
                                    $name.to_string(),
                                    "".to_string()),
                       $name);
        }
    }
    push!(2, "0.1");
    push!(1, "1");
    {
        push!(2, "1.1");
        {
            push!(3, "1.1.1");
            push!(3, "1.1.2");
        }
        push!(2, "1.2");
        {
            push!(3, "1.2.1");
            push!(3, "1.2.2");
        }
    }
    push!(1, "2");
    push!(1, "3");
    {
        push!(4, "3.0.0.1");
        {
            push!(6, "3.0.0.1.0.1");
        }
        push!(4, "3.0.0.2");
        push!(2, "3.1");
        {
            push!(4, "3.1.0.1");
        }
    }

    macro_rules! toc {
        ($(($level: expr, $name: expr, $(($sub: tt))* )),*) => {
            Toc {
                entries: vec![
                    $(
                        TocEntry {
                            level: $level,
                            name: $name.to_string(),
                            sec_number: $name.to_string(),
                            id: "".to_string(),
                            children: toc!($($sub),*)
                        }
                        ),*
                    ]
            }
        }
    }
    let expected = toc!(
        (2, "0.1", ),

        (1, "1",
         ((2, "1.1", ((3, "1.1.1", )) ((3, "1.1.2", ))))
         ((2, "1.2", ((3, "1.2.1", )) ((3, "1.2.2", ))))
         ),

        (1, "2", ),

        (1, "3",
         ((4, "3.0.0.1", ((6, "3.0.0.1.0.1", ))))
         ((4, "3.0.0.2", ))
         ((2, "3.1", ((4, "3.1.0.1", ))))
         )
        );
    assert_eq!(expected, builder.into_toc());
}
