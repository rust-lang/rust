use rustc_serialize::json::{self, Json};
use std::fs::File;
use std::path::Path;

pub struct Mismatch {
    pub path: Vec<String>,
    pub expected: Json,
    pub found: Option<Json>
}

pub fn compare_analysis(output: &Path, expected: &Path) -> Vec<Mismatch> {
    let mut output = json::from_reader(&mut File::open(output).unwrap()).expect("Invalid JSON");
    let expected = json::from_reader(&mut File::open(expected).unwrap()).expect("Invalid JSON");
    check_matching(vec![], &mut output, &expected)
}

// Returns a vector containing mismatches between the keys in `expected` and those
// in `output`. Keys that are present in `output` but not in `expected` are ignored.
fn check_matching(mut path: Vec<String>, output: &mut Json, expected: &Json) -> Vec<Mismatch> {
    match (output, expected) {
        (&mut Json::Object(ref mut o_obj), &Json::Object(ref e_obj)) => {
            let mut mismatches = vec![];

            // Check that all keys in `e_obj` contain the same values as the keys in
            // `o_obj`
            for (k, e_v) in e_obj {
                let mut new_path = path.clone();
                new_path.push(k.clone());

                // If both keys exist, check recursively. Otherwise, signal a mismatch.
                if let Some(o_v) = o_obj.get_mut(k) {
                    mismatches.extend(check_matching(new_path, o_v, &e_v));
                } else {
                    mismatches.push(Mismatch { path: new_path, expected: e_v.clone(), found: None });
                }
            }

            mismatches
        }
        (&mut Json::Array(ref mut o_arr), &Json::Array(ref e_arr)) => {
            let mut mismatches = vec![];
            let mut o_arr = o_arr.clone();
            path.push("<list-element>".to_string());

            // Each element in `e_arr` is compared against each element in `e_obj`,
            // until a match has been found.
            for e in e_arr {
                let mut matches = None;

                // Find a match between `e` and any `o`
                for (i, o) in o_arr.iter_mut().enumerate() {
                    // Note: we can use the empty string as path, since we are not going to use
                    // the mismatches
                    if check_matching(Vec::new(), o, &e).len() == 0 {
                        matches = Some(i);
                        break;
                    }
                }

                if let Some(i) = matches {
                    // Remove the element from `o_arr`, to prevent double matches
                    o_arr.swap_remove(i);
                } else {
                    // No `o` could match `e`
                    mismatches.push(Mismatch { path: path.clone(), expected: e.clone(), found: None });
                }
            }

            mismatches
        }

        // Mismatched types or scalar values
        (output, expected) => {
            if expected == output {
                vec![]
            } else {
                vec![Mismatch {
                    path: path,
                    expected: expected.clone(),
                    found: Some(output.clone())
                }]
            }
        }
    }

    // Enumerate all keys in expected
    // Retrieve the current key from output:
    // * If present, compare values
    // * If absent,
}