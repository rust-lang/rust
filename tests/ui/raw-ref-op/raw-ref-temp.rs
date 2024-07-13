// Ensure that we don't allow taking the address of temporary values
#![feature(type_ascription)]

const FOUR: u64 = 4;

const PAIR: (i32, i64) = (1, 2);

const ARRAY: [i32; 2] = [1, 2];

fn main() {
    let ref_expr = &raw const 2;                                    //~ ERROR cannot take address
    let mut_ref_expr = &raw mut 3;                                  //~ ERROR cannot take address
    let ref_const = &raw const FOUR;                                //~ ERROR cannot take address
    let mut_ref_const = &raw mut FOUR;                              //~ ERROR cannot take address

    let field_ref_expr = &raw const (1, 2).0;                       //~ ERROR cannot take address
    let mut_field_ref_expr = &raw mut (1, 2).0;                     //~ ERROR cannot take address
    let field_ref = &raw const PAIR.0;                              //~ ERROR cannot take address
    let mut_field_ref = &raw mut PAIR.0;                            //~ ERROR cannot take address

    let index_ref_expr = &raw const [1, 2][0];                      //~ ERROR cannot take address
    let mut_index_ref_expr = &raw mut [1, 2][0];                    //~ ERROR cannot take address
    let index_ref = &raw const ARRAY[0];                            //~ ERROR cannot take address
    let mut_index_ref = &raw mut ARRAY[1];                          //~ ERROR cannot take address

    let ref_ascribe = &raw const type_ascribe!(2, i32);             //~ ERROR cannot take address
    let mut_ref_ascribe = &raw mut type_ascribe!(3, i32);           //~ ERROR cannot take address

    let ascribe_field_ref = &raw const type_ascribe!(PAIR.0, i32);  //~ ERROR cannot take address
    let ascribe_index_ref = &raw mut type_ascribe!(ARRAY[0], i32);  //~ ERROR cannot take address
}
