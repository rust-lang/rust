// rustfmt-max_width: 140

impl NotificationRepository {
    fn set_status_changed(
        &self,
        repo_tx_conn: &RepoTxConn,
        rid: &RoutableId,
        changed_at: NaiveDateTime,
    ) -> NukeResult<Option<NotificationStatus>> {
        repo_tx_conn.run(move |conn| {
            let res = diesel::update(client_notification::table)
                .filter(
                    client_notification::routable_id.eq(DieselRoutableId(rid.clone())).and(
                        client_notification::changed_at
                            .lt(changed_at)
                            .or(client_notification::changed_at.is_null()),
                    ),
                )
                .set(client_notification::changed_at.eq(changed_at))
                .returning((
                    client_notification::id,
                    client_notification::changed_at,
                    client_notification::polled_at,
                    client_notification::notified_at,
                ))
                .get_result::<(Uuid, Option<NaiveDateTime>, Option<NaiveDateTime>, Option<NaiveDateTime>)>(conn)
                .optional()?;

            match res {
                Some(row) => {
                    let client_id = client_contract::table
                        .inner_join(client_notification::table)
                        .filter(client_notification::id.eq(row.0))
                        .select(client_contract::client_id)
                        .get_result::<Uuid>(conn)?;

                    Ok(Some(NotificationStatus {
                        client_id: client_id.into(),
                        changed_at: row.1,
                        polled_at: row.2,
                        notified_at: row.3,
                    }))
                }
                None => Ok(None),
            }
        })
    }
}
