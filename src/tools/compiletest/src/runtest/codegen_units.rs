use std::collections::HashSet;

use super::{Emit, TestCx, WillExecute};
use crate::errors;
use crate::util::static_regex;

impl TestCx<'_> {
    pub(super) fn run_codegen_units_test(&self) {
        assert!(self.revision.is_none(), "revisions not relevant here");

        let proc_res = self.compile_test(WillExecute::No, Emit::None);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        self.check_no_compiler_crash(&proc_res, self.props.should_ice);

        const PREFIX: &str = "MONO_ITEM ";
        const CGU_MARKER: &str = "@@";

        // Some MonoItems can contain {closure@/path/to/checkout/tests/codgen-units/test.rs}
        // To prevent the current dir from leaking, we just replace the entire path to the test
        // file with TEST_PATH.
        let actual: Vec<MonoItem> = proc_res
            .stdout
            .lines()
            .filter(|line| line.starts_with(PREFIX))
            .map(|line| line.replace(&self.testpaths.file.as_str(), "TEST_PATH").to_string())
            .map(|line| str_to_mono_item(&line, true))
            .collect();

        let expected: Vec<MonoItem> = errors::load_errors(&self.testpaths.file, None)
            .iter()
            .map(|e| str_to_mono_item(&e.msg[..], false))
            .collect();

        let mut missing = Vec::new();
        let mut wrong_cgus = Vec::new();

        for expected_item in &expected {
            let actual_item_with_same_name = actual.iter().find(|ti| ti.name == expected_item.name);

            if let Some(actual_item) = actual_item_with_same_name {
                if !expected_item.codegen_units.is_empty() &&
                   // Also check for codegen units
                   expected_item.codegen_units != actual_item.codegen_units
                {
                    wrong_cgus.push((expected_item.clone(), actual_item.clone()));
                }
            } else {
                missing.push(expected_item.string.clone());
            }
        }

        let unexpected: Vec<_> = actual
            .iter()
            .filter(|acgu| !expected.iter().any(|ecgu| acgu.name == ecgu.name))
            .map(|acgu| acgu.string.clone())
            .collect();

        if !missing.is_empty() {
            missing.sort();

            println!("\nThese items should have been contained but were not:\n");

            for item in &missing {
                println!("{}", item);
            }

            println!("\n");
        }

        if !unexpected.is_empty() {
            let sorted = {
                let mut sorted = unexpected.clone();
                sorted.sort();
                sorted
            };

            println!("\nThese items were contained but should not have been:\n");

            for item in sorted {
                println!("{}", item);
            }

            println!("\n");
        }

        if !wrong_cgus.is_empty() {
            wrong_cgus.sort_by_key(|pair| pair.0.name.clone());
            println!("\nThe following items were assigned to wrong codegen units:\n");

            for &(ref expected_item, ref actual_item) in &wrong_cgus {
                println!("{}", expected_item.name);
                println!("  expected: {}", codegen_units_to_str(&expected_item.codegen_units));
                println!("  actual:   {}", codegen_units_to_str(&actual_item.codegen_units));
                println!();
            }
        }

        if !(missing.is_empty() && unexpected.is_empty() && wrong_cgus.is_empty()) {
            panic!();
        }

        #[derive(Clone, Eq, PartialEq)]
        struct MonoItem {
            name: String,
            codegen_units: HashSet<String>,
            string: String,
        }

        // [MONO_ITEM] name [@@ (cgu)+]
        fn str_to_mono_item(s: &str, cgu_has_crate_disambiguator: bool) -> MonoItem {
            let s = if s.starts_with(PREFIX) { (&s[PREFIX.len()..]).trim() } else { s.trim() };

            let full_string = format!("{}{}", PREFIX, s);

            let parts: Vec<&str> =
                s.split(CGU_MARKER).map(str::trim).filter(|s| !s.is_empty()).collect();

            let name = parts[0].trim();

            let cgus = if parts.len() > 1 {
                let cgus_str = parts[1];

                cgus_str
                    .split(' ')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        if cgu_has_crate_disambiguator {
                            remove_crate_disambiguators_from_set_of_cgu_names(s)
                        } else {
                            s.to_string()
                        }
                    })
                    .collect()
            } else {
                HashSet::new()
            };

            MonoItem { name: name.to_owned(), codegen_units: cgus, string: full_string }
        }

        fn codegen_units_to_str(cgus: &HashSet<String>) -> String {
            let mut cgus: Vec<_> = cgus.iter().collect();
            cgus.sort();

            let mut string = String::new();
            for cgu in cgus {
                string.push_str(&cgu[..]);
                string.push(' ');
            }

            string
        }

        // Given a cgu-name-prefix of the form <crate-name>.<crate-disambiguator> or
        // the form <crate-name1>.<crate-disambiguator1>-in-<crate-name2>.<crate-disambiguator2>,
        // remove all crate-disambiguators.
        fn remove_crate_disambiguator_from_cgu(cgu: &str) -> String {
            let Some(captures) =
                static_regex!(r"^[^\.]+(?P<d1>\.[[:alnum:]]+)(-in-[^\.]+(?P<d2>\.[[:alnum:]]+))?")
                    .captures(cgu)
            else {
                panic!("invalid cgu name encountered: {cgu}");
            };

            let mut new_name = cgu.to_owned();

            if let Some(d2) = captures.name("d2") {
                new_name.replace_range(d2.start()..d2.end(), "");
            }

            let d1 = captures.name("d1").unwrap();
            new_name.replace_range(d1.start()..d1.end(), "");

            new_name
        }

        // The name of merged CGUs is constructed as the names of the original
        // CGUs joined with "--". This function splits such composite CGU names
        // and handles each component individually.
        fn remove_crate_disambiguators_from_set_of_cgu_names(cgus: &str) -> String {
            cgus.split("--").map(remove_crate_disambiguator_from_cgu).collect::<Vec<_>>().join("--")
        }
    }
}
