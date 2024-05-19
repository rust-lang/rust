This repository was previously licensed under MPL-2.0, however in #3093
([archive](http://web.archive.org/web/20181005185227/https://github.com/rust-lang-nursery/rust-clippy/issues/3093),
[screenshot](https://user-images.githubusercontent.com/1617736/46573505-5b856880-c94b-11e8-9a14-981c889b4981.png)) we
relicensed it to the Rust license (dual licensed as Apache v2 / MIT)

At the time, the contributors were those listed in contributors.txt.

We opened a bunch of issues asking for an explicit relicensing approval. Screenshots of all these issues at the time of
relicensing are archived on GitHub. We also have saved Wayback Machine copies of these:

- #3094
  ([archive](http://web.archive.org/web/20181005191247/https://github.com/rust-lang-nursery/rust-clippy/issues/3094),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573506-5b856880-c94b-11e8-8a44-51cb40bc16ee.png))
- #3095
  ([archive](http://web.archive.org/web/20181005184416/https://github.com/rust-lang-nursery/rust-clippy/issues/3095),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573507-5c1dff00-c94b-11e8-912a-4bd6b5f838f5.png))
- #3096
  ([archive](http://web.archive.org/web/20181005184802/https://github.com/rust-lang-nursery/rust-clippy/issues/3096),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573508-5c1dff00-c94b-11e8-9425-2464f7260ff0.png))
- #3097
  ([archive](http://web.archive.org/web/20181005184821/https://github.com/rust-lang-nursery/rust-clippy/issues/3097),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573509-5c1dff00-c94b-11e8-8ba2-53f687984fe7.png))
- #3098
  ([archive](http://web.archive.org/web/20181005184900/https://github.com/rust-lang-nursery/rust-clippy/issues/3098),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573510-5c1dff00-c94b-11e8-8f64-371698401c60.png))
- #3099
  ([archive](http://web.archive.org/web/20181005184901/https://github.com/rust-lang-nursery/rust-clippy/issues/3099),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573511-5c1dff00-c94b-11e8-8e20-7d0eeb392b95.png))
- #3100
  ([archive](http://web.archive.org/web/20181005184901/https://github.com/rust-lang-nursery/rust-clippy/issues/3100),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573512-5c1dff00-c94b-11e8-8a13-7d758ed3563d.png))
- #3230
  ([archive](http://web.archive.org/web/20181005184903/https://github.com/rust-lang-nursery/rust-clippy/issues/3230),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573513-5cb69580-c94b-11e8-86b1-14ce82741e5c.png))

The usernames of commenters on these issues can be found in relicense_comments.txt

There are a few people in relicense_comments.txt who are not found in contributors.txt:

- @EpocSquadron has [made minor text contributions to the
  README](https://github.com/rust-lang/rust-clippy/commits?author=EpocSquadron) which have since been overwritten, and
  doesn't count
- @JayKickliter [agreed to the relicense on their pull
  request](https://github.com/rust-lang/rust-clippy/pull/3195#issuecomment-423781016)
  ([archive](https://web.archive.org/web/20181005190730/https://github.com/rust-lang/rust-clippy/pull/3195),
  [screenshot](https://user-images.githubusercontent.com/1617736/46573514-5cb69580-c94b-11e8-8ffb-05a5bd02e2cc.png)

- @sanmai-NL's [contribution](https://github.com/rust-lang/rust-clippy/commits?author=sanmai-NL) is a minor one-word
  addition which doesn't count for copyright assignment
- @zmt00's [contributions](https://github.com/rust-lang/rust-clippy/commits?author=zmt00) are minor typo fixes and don't
  count
- @VKlayd has [nonminor contributions](https://github.com/rust-lang/rust-clippy/commits?author=VKlayd) which we rewrote
  (see below)
- @wartman4404 has [nonminor contributions](https://github.com/rust-lang/rust-clippy/commits?author=wartman4404) which
  we rewrote (see below)


Two of these contributors had non-minor contributions (#2184, #427) requiring a rewrite, carried out in #3251
([archive](http://web.archive.org/web/20181005192411/https://github.com/rust-lang-nursery/rust-clippy/pull/3251),
[screenshot](https://user-images.githubusercontent.com/1617736/46573515-5cb69580-c94b-11e8-86e5-b456452121b2.png))

First, I (Manishearth) removed the lints they had added. I then documented at a high level what the lints did in #3251,
asking for co-maintainers who had not seen the code for the lints to rewrite them. #2814 was rewritten by @phansch, and
#427 was rewritten by @oli-obk, who did not recall having previously seen the code they were rewriting.

------

Since this document was written, @JayKickliter and @sanmai-ML added their consent in #3230
([archive](http://web.archive.org/web/20181006171926/https://github.com/rust-lang-nursery/rust-clippy/issues/3230))
