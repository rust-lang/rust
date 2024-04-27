The functional record update syntax was used on something other than a struct.

Erroneous code example:

```compile_fail,E0436
enum PublicationFrequency {
    Weekly,
    SemiMonthly { days: (u8, u8), annual_special: bool },
}

fn one_up_competitor(competitor_frequency: PublicationFrequency)
                     -> PublicationFrequency {
    match competitor_frequency {
        PublicationFrequency::Weekly => PublicationFrequency::SemiMonthly {
            days: (1, 15), annual_special: false
        },
        c @ PublicationFrequency::SemiMonthly{ .. } =>
            PublicationFrequency::SemiMonthly {
                annual_special: true, ..c // error: functional record update
                                          //        syntax requires a struct
        }
    }
}
```

The functional record update syntax is only allowed for structs (struct-like
enum variants don't qualify, for example). To fix the previous code, rewrite the
expression without functional record update syntax:

```
enum PublicationFrequency {
    Weekly,
    SemiMonthly { days: (u8, u8), annual_special: bool },
}

fn one_up_competitor(competitor_frequency: PublicationFrequency)
                     -> PublicationFrequency {
    match competitor_frequency {
        PublicationFrequency::Weekly => PublicationFrequency::SemiMonthly {
            days: (1, 15), annual_special: false
        },
        PublicationFrequency::SemiMonthly{ days, .. } =>
            PublicationFrequency::SemiMonthly {
                days, annual_special: true // ok!
        }
    }
}
```
