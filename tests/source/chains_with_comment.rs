// Chains with comment.

fn main() {
    let x = y // comment
        .z;

    foo // foo
        // comment after parent
        .x
        .y
        // comment 1
        .bar() // comment after bar()
  // comment 2
        .foobar
  // comment after
        // comment 3
        .baz(x, y, z);

    self.rev_dep_graph
        .iter()
       // Remove nodes that are not dirty
        .filter(|&(unit, _)| dirties.contains(&unit))
     // Retain only dirty dependencies of the ones that are dirty
       .map(|(k, deps)| {
            (
                k.clone(),
                deps.iter()
                .cloned()
               .filter(|d| dirties.contains(&d))
              .collect(),
            )
        });

    let y = expr /* comment */.kaas()?
// comment
       .test();
    let loooooooooooooooooooooooooooooooooooooooooong = does_this?.look?.good?.should_we_break?.after_the_first_question_mark?;
    let zzzz = expr?   // comment after parent
// comment 0
.another??? // comment 1
.another????  // comment 2
.another? // comment 3
.another?;

    let y = a.very .loooooooooooooooooooooooooooooooooooooong() /* comment */ .chain()
        .inside()  /* comment */        .weeeeeeeeeeeeeee()? .test()  .0
        .x;

    parameterized(f,
                  substs,
                  def_id,
                  Ns::Value,
                  &[],
                  |tcx| tcx.lookup_item_type(def_id).generics)?;
    fooooooooooooooooooooooooooo()?.bar()?.baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaz()?;

    // #2559
    App::new("cargo-cache")
.version(crate_version!())
.bin_name("cargo")
.about("Manage cargo cache")
.author("matthiaskrgr")
.subcommand(
SubCommand::with_name("cache")
.version(crate_version!())
.bin_name("cargo-cache")
.about("Manage cargo cache")
.author("matthiaskrgr")
.arg(&list_dirs)
.arg(&remove_dir)
.arg(&gc_repos)
.arg(&info)
.arg(&keep_duplicate_crates)    .arg(&dry_run)
.arg(&auto_clean)
.arg(&auto_clean_expensive),
        ) // subcommand
        .arg(&list_dirs);
}

// #2177
impl Foo {
    fn dirty_rev_dep_graph(
        &self,
        dirties: &HashSet<UnitKey>,
    ) -> HashMap<UnitKey, HashSet<UnitKey>> {
        let dirties = self.transitive_dirty_units(dirties);
        trace!("transitive_dirty_units: {:?}", dirties);

        self.rev_dep_graph.iter()
        // Remove nodes that are not dirty
            .filter(|&(unit, _)| dirties.contains(&unit))
        // Retain only dirty dependencies of the ones that are dirty
            .map(|(k, deps)| (k.clone(), deps.iter().cloned().filter(|d| dirties.contains(&d)).collect()))
    }
}

// #2907
fn foo() {
    let x = foo
        .bar??  ? // comment
        .baz;
    let x = foo
        .bar?  ??
    // comment
        .baz;
    let x = foo
        .bar? ? ? // comment
    // comment
        .baz;
    let x = foo
        .bar? ?? // comment
    // comment
        ? ??
    // comment
        ?  ??
    // comment
        ???  
    // comment
        ? ? ?
        .baz;

    parent /* comment */ .child;
    parent /* comment1 */     /* comment2 */ .child;
    parent /* comment1 */   // comment 2
            .child;
    parent /* ???no tries here */ // nor ??? here ????
    ???.child;
    parent/*
        some interesting shaped comment
    */.child;

    parent /* comment */
    // another comment
        .child1(some, args) /* comment */ // again
    .await /* longer
         comment
    */ // more comments!
        .use // here's a comment that reaches right to the default width limit aaaaaaaaaaaaaaaaaaaa
    .end;

    parent? /* spacing is interesting */    ?   /* blah */.child;

    // NB: recording the current state of things
    // but is possibly a bug: rustfmt/issues/6433
    parent //comment
           .1 .2 .3;
}
