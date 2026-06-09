// This test ensures that impl/trait associated items are listed in the sidebar.

// ignore-tidy-linelength

#![feature(inherent_associated_types)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]
#![crate_name = "foo"]

//@ has 'foo/struct.Bar.html'
pub struct Bar;

impl Bar {
    //@ has - '//*[@class="sidebar-elems"]//h3[1]' 'Associated Constants'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block associatedconstant"]/li/a[@href="#associatedconstant.X"]' 'X'
    pub const X: u8 = 12;
    //@ has - '//*[@class="sidebar-elems"]//h3[2]' 'Associated Types'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block associatedtype"]/li/a[@href="#associatedtype.Y"]' 'Y'
    pub type Y = u8;
}

//@ has 'foo/trait.Foo.html'
pub trait Foo {
    //@ has - '//*[@class="sidebar-elems"]//h3[5]' 'Required Methods'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][5]/li/a[@href="#tymethod.yeay"]' 'yeay'
    fn yeay();
    //@ has - '//*[@class="sidebar-elems"]//h3[6]' 'Provided Methods'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][6]/li/a[@href="#method.boo"]' 'boo'
    fn boo() {}
    //@ has - '//*[@class="sidebar-elems"]//h3[1]' 'Required Associated Constants'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][1]/li/a[@href="#associatedconstant.W"]' 'W'
    const W: u32;
    //@ has - '//*[@class="sidebar-elems"]//h3[2]' 'Provided Associated Constants'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][2]/li/a[@href="#associatedconstant.U"]' 'U'
    const U: u32 = 0;
    //@ has - '//*[@class="sidebar-elems"]//h3[3]' 'Required Associated Types'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][3]/li/a[@href="#associatedtype.Z"]' 'Z'
    type Z;
    //@ has - '//*[@class="sidebar-elems"]//h3[4]' 'Provided Associated Types'
    //@ has - '//*[@class="sidebar-elems"]//ul[@class="block"][4]/li/a[@href="#associatedtype.T"]' 'T'
    type T = u32;
}
