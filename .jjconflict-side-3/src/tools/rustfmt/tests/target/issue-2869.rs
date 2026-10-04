// rustfmt-struct_field_align_threshold: 50

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct AuditLog1 {
    creation_time:                     String,
    id:                                String,
    operation:                         String,
    organization_id:                   String,
    record_type:                       u32,
    result_status:                     Option<String>,
    #[serde(rename = "ClientIP")]
    client_ip:                         Option<IpAddr>,
    object_id:                         String,
    actor:                             Option<Vec<IDType>>,
    actor_context_id:                  Option<String>,
    actor_ip_address:                  Option<IpAddr>,
    azure_active_directory_event_type: Option<u8>,

    #[serde(rename = "very")]
    aaaaa: String,
    #[serde(rename = "cool")]
    bb:    i32,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct AuditLog2 {
    creation_time:                     String,
    id:                                String,
    operation:                         String,
    organization_id:                   String,
    record_type:                       u32,
    result_status:                     Option<String>,
    client_ip:                         Option<IpAddr>,
    object_id:                         String,
    actor:                             Option<Vec<IDType>>,
    actor_context_id:                  Option<String>,
    actor_ip_address:                  Option<IpAddr>,
    azure_active_directory_event_type: Option<u8>,
}
