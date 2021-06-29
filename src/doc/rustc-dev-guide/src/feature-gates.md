# Feature Gates

This chapter is intended to provide basic help for adding, removing, and
modifying feature gates.


## Adding a feature gate

See ["Stability in code"] for help with adding a new feature; this section just
covers how to add the feature gate *declaration*.

Add a feature gate declaration to `rustc_feature/src/active.rs` in the active
`declare_features` block:

```rust,ignore
/// description of feature
(active, $feature_name, "$current_nightly_version", Some($tracking_issue_number), $edition)
```

where `$edition` has the type `Option<Edition>`, and is typically
just `None`.

For example:

```rust,ignore
/// Allows defining identifiers beyond ASCII.
(active, non_ascii_idents, "1.0.0", Some(55467), None),
```

Features can be marked as incomplete, and trigger the warn-by-default [`incomplete_features` lint]
by setting their type to `incomplete`:

```rust,ignore
/// Allows unsized rvalues at arguments and parameters.
(incomplete, unsized_locals, "1.30.0", Some(48055), None),
```

When added, the current version should be the one for the current nightly.
Once the feature is moved to `accepted.rs`, the version is changed to that
nightly version.


## Removing a feature gate

[removing]: #removing-a-feature-gate

To remove a feature gate, follow these steps:

1. Remove the feature gate declaration in `rustc_feature/src/active.rs`.
   It will look like this:

   ```rust,ignore
   /// description of feature
   (active, $feature_name, "$version", Some($tracking_issue_number), $edition)
   ```

2. Add a modified version of the feature gate declaration that you just
   removed to `rustc_feature/src/removed.rs`:

   ```rust,ignore
   /// description of feature
   (removed, $old_feature_name, "$version", Some($tracking_issue_number), $edition,
    Some("$why_it_was_removed"))
   ```


## Renaming a feature gate

[renaming]: #renaming-a-feature-gate

To rename a feature gate, follow these steps (the first two are the same steps
to follow when [removing a feature gate][removing]):

1. Remove the old feature gate declaration in `rustc_feature/src/active.rs`.
   It will look like this:

   ```rust,ignore
   /// description of feature
   (active, $old_feature_name, "$version", Some($tracking_issue_number), $edition)
   ```

2. Add a modified version of the old feature gate declaration that you just
   removed to `rustc_feature/src/removed.rs`:

   ```rust,ignore
   /// description of feature
   /// Renamed to `$new_feature_name`
   (removed, $old_feature_name, "$version", Some($tracking_issue_number), $edition,
    Some("renamed to `$new_feature_name`"))
   ```

3. Add a feature gate declaration with the new name to
   `rustc_feature/src/active.rs`. It should look very similar to the old
   declaration:

   ```rust,ignore
   /// description of feature
   (active, $new_feature_name, "$version", Some($tracking_issue_number), $edition)
   ```


## Stabilizing a feature

See ["Updating the feature-gate listing"] in the "Stabilizing Features" chapter
for instructions. There are additional steps you will need to take beyond just
updating the declaration!


["Stability in code"]: ./implementing_new_features.md#stability-in-code
[`incomplete_features` lint]: https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#incomplete-features
["Updating the feature-gate listing"]: ./stabilization_guide.md#updating-the-feature-gate-listing
