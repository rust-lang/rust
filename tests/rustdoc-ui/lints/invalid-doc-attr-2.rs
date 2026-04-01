#![deny(invalid_doc_attributes)]

#![doc("other attribute")]
//~^ ERROR
//~| WARN
#![doc]
//~^ ERROR
