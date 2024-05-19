// ICE: failed to get layout for [type error]
// issue: rust-lang/rust#92979

use std::fs;
use std::fs::File;
use std::io::Read;
use std::convert::TryInto;

fn get_file_as_byte_vec(filename: &String) -> Vec<u8> {
    let mut f = File::open(&filename).expect("no file found");
    let metadata = fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");

    buffer
}



fn demo<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}


fn main() {

    // Specify filepath
    let file: &String = &String::from("SomeBinaryDataFileWith4ByteHeaders_f32s_and_u32s");

    // Read file into a vector of bytes
    let file_data = get_file_as_byte_vec(file);

    // Print length of vector and first few values
    let length = file_data.len();
    println!("The read function read {} bytes", length);
    println!("The first few bytes:");
    for i in 0..20{
        println!("{}", file_data[i]);
    }

    // Manually count just to make sure
    let mut n: u64 = 0;
    for data in file_data{
        n += 1;
    }
    println!("We counted {} bytes", n);
    assert!(n as usize == length, "Manual counting does not equal len method");

    // Simulation parameters
    const N: usize = 49627502;                // Number of Particles
    const bs: f64 = 125.0;                  // Box Size
    const HEADER_INCREMENT: u64 = 4*1;

    // Initialize index and counter variables
    let (mut j, mut pos, mut vel, mut id, mut mass): (u64, u64, u64, u64, u64) = (0, 0, 0, 0, 0);

    // Unpack Position Data
    j += HEADER_INCREMENT;
    let mut position: Vec<f32> = Vec::new();
    while position.len() < N*3 {

        let p: Vec<u8> = Vec::new();
        for item in 0i8..4 {
            let item = item;
            p.push(file_data[j as usize]);
            j += 1;
        }
        &mut position[position.len()] = f32::from_be_bytes(demo(p));
        //~^ ERROR invalid left-hand side of assignment
    }

    // Ensure position data is indeed position by checking values
    for p in position {
        assert!((p > 0.) & (p < 125.), "Not in box")
    }

}
