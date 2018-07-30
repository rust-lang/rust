extern crate itertools;

use std::hash;
use itertools::Itertools;

#[derive(Debug, Eq)]
pub struct Test {
    pub start_line: usize,
    pub name: String,
    pub text: String,
}

impl PartialEq for Test {
    fn eq(&self, other: &Test) -> bool {
        self.name.eq(&other.name)
    }
}

impl hash::Hash for Test {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

pub fn collect_tests(s: &str) -> Vec<Test> {
    let mut res = vec![];
    let prefix = "// ";
    let comment_blocks = s
        .lines()
        .map(str::trim_left)
        .enumerate()
        .group_by(|(_idx, line)| line.starts_with(prefix));

    'outer: for (is_comment, block) in comment_blocks.into_iter() {
        if !is_comment {
            continue;
        }
        let mut block = block.map(|(idx, line)| (idx, &line[prefix.len()..]));

        let (start_line, name) = loop {
            match block.next() {
                Some((idx, line)) if line.starts_with("test ") => {
                    break (idx, line["test ".len()..].to_string())
                },
                Some(_) => (),
                None => continue 'outer,
            }
        };
        let text: String = itertools::join(
            block.map(|(_, line)| line)
                .chain(::std::iter::once("")),
            "\n"
        );
        assert!(!text.trim().is_empty() && text.ends_with("\n"));
        res.push(Test { start_line, name, text })
    }
    res
}
