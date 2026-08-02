#![feature(doc_notable_trait)]
#![deny(invalid_doc_attributes)]

#[doc(notable_trait)]
trait NoColor {}

#[doc(notable_trait())]
trait NoColor2 {}

#[doc(notable_trait(color="transparent"))]
trait Transparent {}

#[doc(notable_trait(color="red"))]
trait Red {}

#[doc(notable_trait="red")]
//~^ ERROR
//~| WARN previously accepted by the compiler
trait InvalidKV {}

#[doc(notable_trait)]
//~^ ERROR
//~| WARN previously accepted by the compiler
struct InvalidNotTrait;

#[doc(notable_trait(check="red"))]
//~^ ERROR
//~| WARN previously accepted by the compiler
trait InvalidKVArg {}

#[doc(notable_trait(color))]
//~^ ERROR
//~| WARN previously accepted by the compiler
trait InvalidAtomArg {}

#[doc(notable_trait(color="invalid_color"))]
//~^ ERROR
//~| WARN previously accepted by the compiler
trait InvalidColor {}

#[doc(notable_trait(color="transparent", color="red"))]
//~^ ERROR
//~| WARN previously accepted by the compiler
trait InvalidMultiple {}
