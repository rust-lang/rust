#![deny(invalid_doc_attributes)]

#![doc("other attribute")]
//~^ ERROR
#![doc]
//~^ ERROR
