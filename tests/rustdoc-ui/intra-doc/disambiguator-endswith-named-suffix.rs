//@ check-pass
//@ normalize-stderr: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"

//! [struct@m!()]   //~ WARN: unmatched disambiguator `struct` and suffix `!()`
//! [struct@m!{}]   //~ WARN: unmatched disambiguator `struct` and suffix `!{}`
//! [struct@m![]]
//! [struct@f()]    //~ WARN: unmatched disambiguator `struct` and suffix `()`
//! [struct@m!]     //~ WARN: unmatched disambiguator `struct` and suffix `!`
//!
//! [enum@m!()]     //~ WARN: unmatched disambiguator `enum` and suffix `!()`
//! [enum@m!{}]     //~ WARN: unmatched disambiguator `enum` and suffix `!{}`
//! [enum@m![]]
//! [enum@f()]      //~ WARN: unmatched disambiguator `enum` and suffix `()`
//! [enum@m!]       //~ WARN: unmatched disambiguator `enum` and suffix `!`
//!
//! [trait@m!()]    //~ WARN: unmatched disambiguator `trait` and suffix `!()`
//! [trait@m!{}]    //~ WARN: unmatched disambiguator `trait` and suffix `!{}`
//! [trait@m![]]
//! [trait@f()]     //~ WARN: unmatched disambiguator `trait` and suffix `()`
//! [trait@m!]      //~ WARN: unmatched disambiguator `trait` and suffix `!`
//!
//! [module@m!()]   //~ WARN: unmatched disambiguator `module` and suffix `!()`
//! [module@m!{}]   //~ WARN: unmatched disambiguator `module` and suffix `!{}`
//! [module@m![]]
//! [module@f()]    //~ WARN: unmatched disambiguator `module` and suffix `()`
//! [module@m!]     //~ WARN: unmatched disambiguator `module` and suffix `!`
//!
//! [mod@m!()]      //~ WARN: unmatched disambiguator `mod` and suffix `!()`
//! [mod@m!{}]      //~ WARN: unmatched disambiguator `mod` and suffix `!{}`
//! [mod@m![]]
//! [mod@f()]       //~ WARN: unmatched disambiguator `mod` and suffix `()`
//! [mod@m!]        //~ WARN: unmatched disambiguator `mod` and suffix `!`
//!
//! [const@m!()]    //~ WARN: unmatched disambiguator `const` and suffix `!()`
//! [const@m!{}]    //~ WARN: unmatched disambiguator `const` and suffix `!{}`
//! [const@m![]]
//! [const@f()]     //~ WARN: incompatible link kind for `f`
//! [const@m!]      //~ WARN: unmatched disambiguator `const` and suffix `!`
//!
//! [constant@m!()]   //~ WARN: unmatched disambiguator `constant` and suffix `!()`
//! [constant@m!{}]   //~ WARN:  unmatched disambiguator `constant` and suffix `!{}`
//! [constant@m![]]
//! [constant@f()]    //~ WARN: incompatible link kind for `f`
//! [constant@m!]     //~ WARN: unmatched disambiguator `constant` and suffix `!`
//!
//! [static@m!()]   //~ WARN: unmatched disambiguator `static` and suffix `!()`
//! [static@m!{}]   //~ WARN: unmatched disambiguator `static` and suffix `!{}`
//! [static@m![]]
//! [static@f()]    //~ WARN: incompatible link kind for `f`
//! [static@m!]     //~ WARN: unmatched disambiguator `static` and suffix `!`
//!
//! [function@m!()]   //~ WARN: unmatched disambiguator `function` and suffix `!()`
//! [function@m!{}]   //~ WARN: unmatched disambiguator `function` and suffix `!{}`
//! [function@m![]]
//! [function@f()]
//! [function@m!]     //~ WARN: unmatched disambiguator `function` and suffix `!`
//!
//! [fn@m!()]   //~ WARN: unmatched disambiguator `fn` and suffix `!()`
//! [fn@m!{}]   //~ WARN: unmatched disambiguator `fn` and suffix `!{}`
//! [fn@m![]]
//! [fn@f()]
//! [fn@m!]     //~ WARN: unmatched disambiguator `fn` and suffix `!`
//!
//! [method@m!()]   //~ WARN: unmatched disambiguator `method` and suffix `!()`
//! [method@m!{}]   //~ WARN: unmatched disambiguator `method` and suffix `!{}`
//! [method@m![]]
//! [method@f()]
//! [method@m!]     //~ WARN: unmatched disambiguator `method` and suffix `!`
//!
//! [derive@m!()]   //~ WARN: incompatible link kind for `m`
//! [derive@m!{}]   //~ WARN: incompatible link kind for `m`
//! [derive@m![]]
//! [derive@f()]    //~ WARN: unmatched disambiguator `derive` and suffix `()`
//! [derive@m!]     //~ WARN: incompatible link kind for `m`
//!
//! [type@m!()]   //~ WARN: unmatched disambiguator `type` and suffix `!()`
//! [type@m!{}]   //~ WARN: unmatched disambiguator `type` and suffix `!{}`
//! [type@m![]]
//! [type@f()]    //~ WARN: unmatched disambiguator `type` and suffix `()`
//! [type@m!]     //~ WARN: unmatched disambiguator `type` and suffix `!`
//!
//! [value@m!()]   //~ WARN: unmatched disambiguator `value` and suffix `!()`
//! [value@m!{}]   //~ WARN: unmatched disambiguator `value` and suffix `!{}`
//! [value@m![]]
//! [value@f()]
//! [value@m!]     //~ WARN: unmatched disambiguator `value` and suffix `!`
//!
//! [macro@m!()]
//! [macro@m!{}]
//! [macro@m![]]
//! [macro@f()]    //~ WARN: unmatched disambiguator `macro` and suffix `()`
//! [macro@m!]
//!
//! [prim@m!()]   //~ WARN: unmatched disambiguator `prim` and suffix `!()`
//! [prim@m!{}]   //~ WARN: unmatched disambiguator `prim` and suffix `!{}`
//! [prim@m![]]
//! [prim@f()]    //~ WARN: unmatched disambiguator `prim` and suffix `()`
//! [prim@m!]     //~ WARN: unmatched disambiguator `prim` and suffix `!`
//!
//! [primitive@m!()]   //~ WARN: unmatched disambiguator `primitive` and suffix `!()`
//! [primitive@m!{}]   //~ WARN: unmatched disambiguator `primitive` and suffix `!{}`
//! [primitive@m![]]
//! [primitive@f()]    //~ WARN: unmatched disambiguator `primitive` and suffix `()`
//! [primitive@m!]     //~ WARN: unmatched disambiguator `primitive` and suffix `!`

#[macro_export]
macro_rules! m {
    () => {};
}

pub fn f() {}
