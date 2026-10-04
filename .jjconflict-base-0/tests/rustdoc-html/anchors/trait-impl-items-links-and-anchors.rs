pub trait MyTrait {
    type Assoc;
    const VALUE: u32 = 12;
    fn trait_function(&self);
    fn defaulted(&self) {}
    fn defaulted_override(&self) {}
}

impl MyTrait for String {
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedtype.Assoc-1"]//a[@class="associatedtype"]/@href' #associatedtype.Assoc
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedtype.Assoc-1"]//a[@class="anchor"]/@href' #associatedtype.Assoc-1
    type Assoc = ();
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedconstant.VALUE-1"]//a[@class="constant"]/@href' #associatedconstant.VALUE
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedconstant.VALUE-1"]//a[@class="anchor"]/@href' #associatedconstant.VALUE-1
    const VALUE: u32 = 5;
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.trait_function"]//a[@class="fn"]/@href' #tymethod.trait_function
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.trait_function"]//a[@class="anchor"]/@href' #method.trait_function
    fn trait_function(&self) {}
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.defaulted_override-1"]//a[@class="fn"]/@href' #method.defaulted_override
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.defaulted_override-1"]//a[@class="anchor"]/@href' #method.defaulted_override-1
    fn defaulted_override(&self) {}
}

impl MyTrait for Vec<u8> {
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedtype.Assoc-2"]//a[@class="associatedtype"]/@href' #associatedtype.Assoc
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedtype.Assoc-2"]//a[@class="anchor"]/@href' #associatedtype.Assoc-2
    type Assoc = ();
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedconstant.VALUE-2"]//a[@class="constant"]/@href' #associatedconstant.VALUE
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedconstant.VALUE-2"]//a[@class="anchor"]/@href' #associatedconstant.VALUE-2
    const VALUE: u32 = 5;
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.trait_function"]//a[@class="fn"]/@href' #tymethod.trait_function
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.trait_function-1"]//a[@class="anchor"]/@href' #method.trait_function-1
    fn trait_function(&self) {}
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.defaulted_override-2"]//a[@class="fn"]/@href' #method.defaulted_override
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="method.defaulted_override-2"]//a[@class="anchor"]/@href' #method.defaulted_override-2
    fn defaulted_override(&self) {}
}

impl MyTrait for MyStruct {
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedtype.Assoc-3"]//a[@class="anchor"]/@href' #associatedtype.Assoc-3
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="associatedtype.Assoc"]//a[@class="associatedtype"]/@href' trait.MyTrait.html#associatedtype.Assoc
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="associatedtype.Assoc"]//a[@class="anchor"]/@href' #associatedtype.Assoc
    type Assoc = bool;
    //@ has trait_impl_items_links_and_anchors/trait.MyTrait.html '//*[@id="associatedconstant.VALUE-3"]//a[@class="anchor"]/@href' #associatedconstant.VALUE-3
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="associatedconstant.VALUE"]//a[@class="constant"]/@href' trait.MyTrait.html#associatedconstant.VALUE
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="associatedconstant.VALUE"]//a[@class="anchor"]/@href' #associatedconstant.VALUE
    const VALUE: u32 = 20;
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.trait_function"]//a[@class="fn"]/@href' trait.MyTrait.html#tymethod.trait_function
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.trait_function"]//a[@class="anchor"]/@href' #method.trait_function
    fn trait_function(&self) {}
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.defaulted_override"]//a[@class="fn"]/@href' trait.MyTrait.html#method.defaulted_override
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.defaulted_override"]//a[@class="anchor"]/@href' #method.defaulted_override
    fn defaulted_override(&self) {}
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.defaulted"]//a[@class="fn"]/@href' trait.MyTrait.html#method.defaulted
    //@ has trait_impl_items_links_and_anchors/struct.MyStruct.html '//*[@id="method.defaulted"]//a[@class="anchor"]/@href' #method.defaulted
}

pub struct MyStruct;

// We check that associated items with default values aren't generated in the implementors list.
impl MyTrait for (u8, u8) {
    //@ !has trait_impl_items_links_and_anchors/trait.MyTrait.html '//div[@id="associatedconstant.VALUE-4"]' ''
    type Assoc = bool;
    fn trait_function(&self) {}
}
