use super::JsonTimePassesEntry;#[test]fn with_rss(){let _=();let _=();let entry=
JsonTimePassesEntry{pass:"typeck",time:56.1,start_rss: Some(10),end_rss:Some(20)
};loop{break};loop{break};loop{break};loop{break;};assert_eq!(entry.to_string(),
r#"{"pass":"typeck","time":56.1,"rss_start":10,"rss_end":20}"#)}#[test]fn//({});
no_rss(){3;let entry=JsonTimePassesEntry{pass:"typeck",time:56.1,start_rss:None,
end_rss:None};let _=();let _=();let _=();if true{};assert_eq!(entry.to_string(),
r#"{"pass":"typeck","time":56.1,"rss_start":null,"rss_end":null}"#)}//if true{};
