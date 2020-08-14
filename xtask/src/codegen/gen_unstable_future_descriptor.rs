//! Generates descriptors structure for unstable feature from Unstable Book

use crate::codegen::update;
use crate::codegen::{self, project_root, Mode, Result};
use chrono::prelude::*;
use fs::read_to_string;
use graphql_client::*;
use proc_macro2::TokenStream;
use quote::quote;
use regex::Regex;
use reqwest;
use serde::*;
use std::fs;
use std::fs::File;
use std::io::copy;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::Command;
use walkdir::WalkDir;

type URI = String;
type DateTime = String;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "src/codegen/schema.graphql",
    query_path = "src/codegen/last_commit_that_affected_path.graphql"
)]
struct LastCommitThatAffectedPath;

fn deep_destructuring(
    response_body: Response<last_commit_that_affected_path::ResponseData>,
) -> CommitInfo {
    let t = response_body.data.unwrap().repository.unwrap().object.unwrap().on;

    use last_commit_that_affected_path::LastCommitThatAffectedPathRepositoryObjectOn::Commit;
    let commit = match t {
        Commit(data) => data,
        _ => panic!("type does not match"),
    };
    let edges = commit.history.edges.unwrap();
    let node = edges.first().unwrap().as_ref().unwrap().node.as_ref().unwrap();
    CommitInfo { commit_url: node.commit_url.clone(), committed_date: node.committed_date.clone() }
}

struct CommitInfo {
    commit_url: String,
    committed_date: String,
}

fn last_update(
    owner: &str,
    name: &str,
    path: &str,
    auth_token: Option<&str>,
) -> Result<CommitInfo> {
    let query =
        LastCommitThatAffectedPath::build_query(last_commit_that_affected_path::Variables {
            owner: owner.to_owned(),
            name: name.to_owned(),
            path: path.to_owned(),
        });

    let client = reqwest::blocking::Client::new();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("User-Agent", "https://github.com/rust-analyzer/rust-analyzer".parse()?);
    let mut request = client.post("https://api.github.com/graphql").headers(headers).json(&query);

    if auth_token.is_some() {
        request = request.bearer_auth(auth_token.unwrap());
    }

    let response = request.send()?;

    let response_body: Response<last_commit_that_affected_path::ResponseData> = response.json()?;
    Ok(deep_destructuring(response_body))
}

fn generate_descriptor(src_dir: PathBuf) -> Result<TokenStream> {
    let files = WalkDir::new(src_dir.join("language-features"))
        .into_iter()
        .chain(WalkDir::new(src_dir.join("library-features")))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file()
                && entry.path().extension().map(|ext| ext == "md").unwrap_or(false)
        })
        .collect::<Vec<_>>();

    let definitions = files
        .iter()
        .map(|entry| {
            let path = entry.path();
            let feature_ident =
                format!("{}", path.file_stem().unwrap().to_str().unwrap().replace("-", "_"));
            let doc = format!("{}", std::fs::read_to_string(path).unwrap());

            quote! { LintCompletion { label: #feature_ident, description: #doc } }
        })
        .collect::<Vec<_>>();

    let ts = quote! {
        use crate::completion::LintCompletion;

        pub const UNSTABLE_FEATURE_DESCRIPTOR:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };
    Ok(ts)
}

fn add_anchor(text: impl std::fmt::Display, anchor: &str) -> String {
    let anchor_str = format!(
        r#"//The anchor is used to check if file is up to date and represent the time 
    //of the last commit that affected path where located data for generation
    //ANCHOR: {}"#,
        anchor
    );
    format!("{}\n\n{}\n", anchor_str, text)
}

fn is_actual(path: &PathBuf, str_datetime: &str) -> bool {
    let re = Regex::new(r"//ANCHOR: (\S*)").unwrap();
    let opt_str = fs::read_to_string(path);
    if opt_str.is_err() {
        return false;
    }
    let text = opt_str.unwrap();
    let opt_datetime = re.captures(text.as_str());
    if opt_datetime.is_none() {
        return false;
    }
    let str_file_dt = opt_datetime.unwrap().get(1).unwrap().as_str();
    let file_dt = str_file_dt.parse::<chrono::DateTime<Utc>>().unwrap();
    let datetime = str_datetime.parse::<chrono::DateTime<Utc>>().unwrap();

    file_dt == datetime
}

fn download_tar(
    owner: &str,
    name: &str,
    auth_token: Option<&str>,
    destination: &PathBuf,
    fname: &str,
) -> Result<()> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("User-Agent", "https://github.com/rust-analyzer/rust-analyzer".parse()?);
    let mut request = reqwest::blocking::Client::new()
        .get(format!("https://api.github.com/repos/{}/{}/tarball", owner, name).as_str())
        .headers(headers);

    if auth_token.is_some() {
        request = request.bearer_auth(auth_token.unwrap());
    }

    let response = request.send()?;
    let download_url = response.url();
    
    let mut response = reqwest::blocking::Client::new()
        .get(download_url.as_str())
        .send()?;

    let mut n = fname.to_string();
    n.push_str(".tar.gz");
    let fpath = destination.join(n);
    let mut file = File::create(fpath)?; 
    response.copy_to(&mut file)?;

    Ok(())
}

pub fn generate_unstable_future_descriptor(mode: Mode) -> Result<()> {
    const auth_token: Option<&str> = None;

    let path = project_root().join(codegen::STORAGE);
    fs::create_dir_all(path.clone())?;

    let commit_info =
        last_update(codegen::REPO_OWNER, codegen::REPO_NAME, codegen::REPO_PATH, auth_token)?;

    if is_actual(
        &project_root().join(codegen::GENERATION_DESTINATION),
        commit_info.committed_date.as_str(),
    ) {
        return Ok(());
    } 

    download_tar(codegen::REPO_OWNER, codegen::REPO_NAME, auth_token, &path, "repository")?;
    Command::new("tar")
        .args(&["-xvf", concat!("repository",".tar.gz"), "--wildcards", "*/src/doc/unstable-book/src", "--strip=1"])
        .current_dir(codegen::STORAGE)
        .output()?;

    let src_dir = path.join(codegen::REPO_PATH);
    let gen_holder = generate_descriptor(src_dir)?.to_string();
    let gen_holder = add_anchor(gen_holder, commit_info.committed_date.as_str());

    let destination = project_root().join(codegen::GENERATION_DESTINATION);
    let contents = crate::reformat(gen_holder)?;
    update(destination.as_path(), &contents, mode)?;

    Ok(())
}
