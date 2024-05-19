// issue: #110623
//@ check-pass

use std::{collections::BTreeMap, num::ParseIntError, str::FromStr};

enum FileSystem {
    File(usize),
    Directory(BTreeMap<String, FileSystem>),
}

impl FromStr for FileSystem {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("dir") {
            Ok(Self::new_dir())
        } else {
            Ok(Self::File(s.split_whitespace().next().unwrap().parse()?))
        }
    }
}

impl FileSystem {
    fn new_dir() -> FileSystem {
        FileSystem::Directory(BTreeMap::new())
    }

    fn insert(&mut self, name: String, other: FileSystem) -> Option<FileSystem> {
        match self {
            FileSystem::File(_) => panic!("can only insert into directory!"),
            FileSystem::Directory(tree) => tree.insert(name, other),
        }
    }

    // Recursively build a tree from commands. This uses (abuses?)
    // the fact that `cd /` only appears at the start and that
    // subsequent `cd`s can only move ONE level to use the recursion
    // stack as the filesystem stack
    fn build<'a>(
        &mut self,
        mut commands: impl Iterator<Item = &'a str> + 'a,
    ) -> Option<impl Iterator<Item = &'a str> + 'a> {
        let cmd = commands.next()?;
        let mut elements = cmd.lines();
        match elements.next().map(str::trim) {
            Some("cd /") | None => self.build(commands),
            Some("cd ..") => {
                // return to higher scope
                Some(commands)
            }
            Some("ls") => {
                for item in elements {
                    let name = item.split_whitespace().last().unwrap();
                    let prior = self.insert(name.to_string(), item.parse().unwrap());
                    debug_assert!(prior.is_none());
                }
                // continue on
                self.build(commands)
            }
            Some(other_cd) => {
                let name = other_cd
                    .trim()
                    .strip_prefix("cd ")
                    .expect("expected a cd command");
                let mut directory = FileSystem::new_dir();
                let further_commands = directory.build(commands);
                self.insert(name.to_string(), directory);
                self.build(further_commands?) // THIS LINE FAILS TO COMPILE
            }
        }
    }
}

fn main() {}
